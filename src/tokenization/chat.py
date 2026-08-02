"""Chat templating and loss masking for instruction and reasoning data.

A chat template turns a list of ``{"role": ..., "content": ...}`` messages into
a single token sequence, and - just as importantly - tells the trainer which
positions should contribute to the loss. Training on the prompt as well as the
answer is one of the most common and most damaging SFT bugs: the model spends
capacity learning to predict user turns it will never need to generate.

The default template is deliberately minimal::

    <|system|>You are helpful.<|end|>
    <|user|>Hi<|end|>
    <|assistant|>Hello!<|end|>

Reasoning traces are wrapped in ``<|think|> ... <|/think|>`` inside the
assistant turn, so a chain-of-thought model can be trained and then have its
reasoning suppressed at inference by stopping generation at ``<|/think|>`` or by
hiding that span.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from minimodel.tokenization.tokenize import BPETokenizer

__all__ = [
    "ChatTemplate",
    "Message",
    "RenderedChat",
    "normalize_messages",
]

#: A single chat message.
Message = dict[str, Any]

#: Loss-mask sentinel for positions the model should not be trained on.
IGNORE_INDEX = -100


@dataclass
class RenderedChat:
    """Token ids for a conversation plus the labels used for training."""

    input_ids: list[int]
    labels: list[int]
    text: str

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def n_supervised(self) -> int:
        """How many positions actually contribute to the loss."""
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def normalize_messages(raw: Any) -> list[Message]:
    """Coerce common dataset shapes into a list of role/content messages.

    Handles the three layouts that cover almost every public SFT dataset:

    * already-normalised ``[{"role": ..., "content": ...}]``
    * Alpaca-style ``{"instruction": ..., "input": ..., "output": ...}``
    * simple pair fields such as ``{"prompt": ..., "response": ...}``

    >>> normalize_messages({"instruction": "Hi", "output": "Hello"})
    [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]
    """
    if isinstance(raw, list):
        messages: list[Message] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("from") or "user"
            content = item.get("content")
            if content is None:
                content = item.get("value", "")
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(str(role), str(role))
            message: Message = {"role": role, "content": str(content)}
            reasoning = item.get("reasoning") or item.get("thinking")
            if reasoning:
                message["reasoning"] = str(reasoning)
            messages.append(message)
        return messages

    if not isinstance(raw, dict):
        raise TypeError(f"cannot interpret {type(raw).__name__} as chat messages")

    if "messages" in raw:
        return normalize_messages(raw["messages"])
    if "conversations" in raw:
        return normalize_messages(raw["conversations"])

    messages = []
    system = raw.get("system") or raw.get("system_prompt")
    if system:
        messages.append({"role": "system", "content": str(system)})

    instruction = raw.get("instruction") or raw.get("prompt") or raw.get("question") or raw.get("input")
    extra_input = raw.get("input") if raw.get("instruction") else None
    if instruction is None:
        raise ValueError(f"no recognisable prompt field in record with keys {sorted(raw)}")
    user_content = str(instruction)
    if extra_input:
        user_content = f"{user_content}\n\n{extra_input}"
    messages.append({"role": "user", "content": user_content})

    reasoning = raw.get("reasoning") or raw.get("thinking") or raw.get("chain_of_thought")
    answer = (
        raw.get("output")
        if raw.get("output") is not None
        else raw.get("response")
        if raw.get("response") is not None
        else raw.get("answer")
        if raw.get("answer") is not None
        else raw.get("completion")
    )
    if answer is None:
        raise ValueError(f"no recognisable answer field in record with keys {sorted(raw)}")
    message: Message = {"role": "assistant", "content": str(answer)}
    if reasoning:
        message["reasoning"] = str(reasoning)
    messages.append(message)
    return messages


@dataclass
class ChatTemplate:
    """Renders conversations to tokens with an assistant-only loss mask.

    Parameters
    ----------
    tokenizer:
        Tokenizer providing the role marker ids.
    train_on_prompt:
        Supervise user/system turns too. Off by default.
    reasoning_weight_marker:
        When a message carries a ``reasoning`` field it is emitted inside
        ``<|think|>``/``<|/think|>``. Set ``supervise_reasoning=False`` to train
        only on the final answer, which is useful when distilling long traces
        into a model too small to reproduce them.
    """

    tokenizer: BPETokenizer
    train_on_prompt: bool = False
    supervise_reasoning: bool = True
    add_bos: bool = True
    add_eos: bool = True
    default_system: str | None = None

    ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
    END_TOKEN = "<|end|>"
    THINK_OPEN = "<|think|>"
    THINK_CLOSE = "<|/think|>"

    def _special(self, token: str) -> list[int]:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            # Tokenizers trained without the chat markers still work; the marker
            # simply becomes ordinary text.
            return self.tokenizer.encode(token, allow_special=False)
        return [token_id]

    def render(
        self,
        messages: Sequence[Message] | dict[str, Any],
        *,
        add_generation_prompt: bool = False,
    ) -> RenderedChat:
        """Render ``messages`` to ids and labels.

        Parameters
        ----------
        add_generation_prompt:
            Append a trailing ``<|assistant|>`` marker so the model continues
            from an assistant turn. Used at inference, not during training.
        """
        normalized = normalize_messages(messages)
        if self.default_system and not any(m["role"] == "system" for m in normalized):
            normalized = [{"role": "system", "content": self.default_system}, *normalized]

        input_ids: list[int] = []
        labels: list[int] = []
        text_parts: list[str] = []

        def emit(ids: Sequence[int], *, supervised: bool) -> None:
            input_ids.extend(ids)
            labels.extend(ids if supervised else [IGNORE_INDEX] * len(ids))

        if self.add_bos:
            emit([self.tokenizer.bos_id], supervised=False)

        for message in normalized:
            role = message["role"]
            marker = self.ROLE_TOKENS.get(role, self.ROLE_TOKENS["user"])
            is_assistant = role == "assistant"
            supervised = is_assistant or self.train_on_prompt

            # The role marker itself is never supervised: the model does not
            # choose who speaks next, the harness does.
            emit(self._special(marker), supervised=False)
            text_parts.append(marker)

            reasoning = message.get("reasoning")
            if is_assistant and reasoning:
                emit(self._special(self.THINK_OPEN), supervised=supervised)
                emit(
                    self.tokenizer.encode(str(reasoning), allow_special=False),
                    supervised=supervised and self.supervise_reasoning,
                )
                emit(self._special(self.THINK_CLOSE), supervised=supervised)
                text_parts.append(f"{self.THINK_OPEN}{reasoning}{self.THINK_CLOSE}")

            content_ids = self.tokenizer.encode(str(message["content"]), allow_special=False)
            emit(content_ids, supervised=supervised)
            text_parts.append(str(message["content"]))

            emit(self._special(self.END_TOKEN), supervised=supervised)
            text_parts.append(self.END_TOKEN)

        if add_generation_prompt:
            emit(self._special(self.ROLE_TOKENS["assistant"]), supervised=False)
            text_parts.append(self.ROLE_TOKENS["assistant"])
        elif self.add_eos:
            emit([self.tokenizer.eos_id], supervised=True)

        return RenderedChat(input_ids=input_ids, labels=labels, text="".join(text_parts))

    def render_prompt(self, messages: Sequence[Message] | dict[str, Any]) -> list[int]:
        """Render only the prompt, ending with the assistant marker."""
        normalized = normalize_messages(messages)
        prompt_only = [m for m in normalized if m["role"] != "assistant"]
        return self.render(prompt_only, add_generation_prompt=True).input_ids

    def stop_token_ids(self) -> list[int]:
        """Token ids that should terminate generation for this template."""
        stops = []
        for token in (self.END_TOKEN, "<|endoftext|>"):
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                stops.append(token_id)
        return stops
