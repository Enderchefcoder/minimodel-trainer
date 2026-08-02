"""User-facing inference: loading a model, chatting, batch generation.

A "model directory" here is anything with ``config.json`` and ``model.pt``,
optionally alongside ``tokenizer.json`` - which is what both
:meth:`~minimodel.architectures.base.BaseLanguageModel.save_pretrained` and
:meth:`~minimodel.checkpointing.CheckpointManager.export_model` produce.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minimodel.architectures.base import BaseLanguageModel
from minimodel.architectures.builder import load_model
from minimodel.core.devices import resolve_device, resolve_dtype
from minimodel.core.logging_utils import get_logger
from minimodel.inference.sampling import (
    SamplingConfig,
    generate,
    generate_batch,
    stream_generate,
)
from minimodel.tokenization.chat import ChatTemplate
from minimodel.tokenization.tokenize import BPETokenizer

__all__ = [
    "LoadedModel",
    "chat_loop",
    "complete",
    "generate_with_reasoning",
    "load_for_inference",
]

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """A model plus its tokenizer and chat template, ready to generate."""

    model: BaseLanguageModel
    tokenizer: BPETokenizer
    template: ChatTemplate
    device: torch.device
    path: Path

    @property
    def parameters(self) -> int:
        """Parameter count, for display."""
        return self.model.num_parameters()

    def __repr__(self) -> str:
        return (
            f"LoadedModel({self.model.architecture_name}, {self.parameters:,} params, "
            f"vocab={self.tokenizer.vocab_size}, device={self.device})"
        )


def load_for_inference(
    path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    device: str | torch.device | None = "auto",
    dtype: str | torch.dtype | None = None,
    system_prompt: str | None = None,
) -> LoadedModel:
    """Load a model directory for generation.

    The tokenizer is looked up next to the model, then one directory up, before
    falling back to ``tokenizer_path``. That covers both a self-contained bundle
    and a run directory where the tokenizer lives beside the checkpoints.
    """
    path = Path(path)
    resolved_device = resolve_device(device)
    model = load_model(path, device=resolved_device)
    model.eval()
    if dtype is not None:
        model = model.to(resolve_dtype(dtype, resolved_device))

    candidates = [
        Path(tokenizer_path) if tokenizer_path else None,
        path / "tokenizer.json",
        path.parent / "tokenizer.json",
        path.parent.parent / "tokenizer.json",
    ]
    tokenizer = None
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            tokenizer = BPETokenizer.load(candidate)
            break
    if tokenizer is None:
        raise FileNotFoundError(
            f"no tokenizer.json found near {path}; pass tokenizer_path explicitly"
        )

    template = ChatTemplate(tokenizer, default_system=system_prompt)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=resolved_device,
        path=path,
    )


def complete(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    seed: int | None = None,
    chat: bool = False,
    include_prompt: bool = False,
    **model_kwargs: Any,
) -> str:
    """Complete a prompt.

    With ``chat=True`` the prompt is wrapped in the chat template and generation
    stops at the end-of-turn marker; otherwise it is treated as raw text for a
    base model to continue.
    """
    config = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_token_ids=loaded.template.stop_token_ids() if chat else [],
        model_kwargs=dict(model_kwargs),
    )
    if chat:
        prompt_ids = loaded.template.render_prompt([{"role": "user", "content": prompt}])
    else:
        prompt_ids = loaded.tokenizer.encode(prompt, add_bos=True)

    output = generate(
        loaded.model,
        torch.tensor([prompt_ids], dtype=torch.long),
        config,
        device=loaded.device,
    )
    ids = output[0].tolist()
    if not include_prompt:
        ids = ids[len(prompt_ids) :]
    return loaded.tokenizer.decode(ids)


def complete_batch(
    loaded: LoadedModel,
    prompts: Sequence[str],
    *,
    config: SamplingConfig | None = None,
) -> list[str]:
    """Complete several prompts in one batched decoding loop."""
    return generate_batch(loaded.model, loaded.tokenizer, prompts, config, device=loaded.device)


def generate_with_reasoning(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_reasoning_tokens: int = 256,
    max_answer_tokens: int = 128,
    temperature: float = 0.7,
    reasoning_temperature: float | None = None,
    **model_kwargs: Any,
) -> dict[str, str]:
    """Generate a reasoning trace and an answer separately.

    The two phases are decoded with separate budgets and separate temperatures.
    That is not cosmetic: a small model left to decide for itself when to stop
    reasoning frequently never stops, and a hard budget on the trace is the
    difference between a usable model and one that times out. A slightly higher
    temperature during reasoning and a lower one for the answer also tends to
    help - exploration while thinking, determinism when committing.

    Returns a dict with ``reasoning``, ``answer`` and ``full`` keys.
    """
    tokenizer = loaded.tokenizer
    template = loaded.template
    think_close = tokenizer.token_to_id(template.THINK_CLOSE)

    prompt_ids = template.render_prompt([{"role": "user", "content": prompt}])
    think_open = tokenizer.token_to_id(template.THINK_OPEN)
    if think_open is not None:
        prompt_ids = [*prompt_ids, think_open]

    reasoning_config = SamplingConfig(
        max_new_tokens=max_reasoning_tokens,
        temperature=reasoning_temperature if reasoning_temperature is not None else temperature,
        top_p=0.95,
        stop_token_ids=[t for t in (think_close, tokenizer.eos_id) if t is not None],
        model_kwargs=dict(model_kwargs),
    )
    reasoning_output = generate(
        loaded.model,
        torch.tensor([prompt_ids], dtype=torch.long),
        reasoning_config,
        device=loaded.device,
    )
    reasoning_ids = reasoning_output[0].tolist()[len(prompt_ids) :]
    reasoning_text = tokenizer.decode(reasoning_ids)

    # Close the thought explicitly so the answer phase starts from a
    # well-formed prefix even when the budget cut the trace short.
    continuation = reasoning_output[0].tolist()
    if think_close is not None and (not reasoning_ids or reasoning_ids[-1] != think_close):
        continuation = [*continuation, think_close]

    answer_config = SamplingConfig(
        max_new_tokens=max_answer_tokens,
        temperature=max(0.0, temperature - 0.2),
        top_p=0.9,
        stop_token_ids=template.stop_token_ids(),
        model_kwargs=dict(model_kwargs),
    )
    answer_output = generate(
        loaded.model,
        torch.tensor([continuation], dtype=torch.long),
        answer_config,
        device=loaded.device,
    )
    answer_text = tokenizer.decode(answer_output[0].tolist()[len(continuation) :])

    return {
        "reasoning": reasoning_text.strip(),
        "answer": answer_text.strip(),
        "full": tokenizer.decode(answer_output[0].tolist()),
    }


def stream_completion(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    chat: bool = True,
    **model_kwargs: Any,
) -> Iterator[str]:
    """Yield decoded text incrementally.

    Tokens are buffered until they form valid UTF-8, so multi-byte characters
    are never emitted half-finished.
    """
    tokenizer = loaded.tokenizer
    if chat:
        prompt_ids = loaded.template.render_prompt([{"role": "user", "content": prompt}])
        stops = loaded.template.stop_token_ids()
    else:
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        stops = [tokenizer.eos_id]

    config = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stops,
        model_kwargs=dict(model_kwargs),
    )
    pending: list[int] = []
    for token_id in stream_generate(
        loaded.model,
        torch.tensor([prompt_ids], dtype=torch.long),
        config,
        device=loaded.device,
    ):
        pending.append(token_id)
        text = tokenizer.decode(pending)
        if "\ufffd" not in text:
            yield text
            pending = []
    if pending:
        yield tokenizer.decode(pending)


def chat_loop(
    loaded: LoadedModel,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    system_prompt: str | None = None,
    stream: bool = True,
    max_history_turns: int = 8,
    **model_kwargs: Any,
) -> None:
    """Run an interactive terminal chat.

    History is trimmed to the last ``max_history_turns`` exchanges, because the
    models this toolkit trains have short context windows and a long history
    silently pushes the actual question out of view.
    """
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    print(f"{loaded.model.architecture_name} - {loaded.parameters:,} parameters")
    print("Type /reset to clear history, /exit to quit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return
        if user_input == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})
        trimmed = history[-(max_history_turns * 2 + 1) :]
        prompt_ids = loaded.template.render_prompt(trimmed)

        config = SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            stop_token_ids=loaded.template.stop_token_ids(),
            model_kwargs=dict(model_kwargs),
        )

        started = time.perf_counter()
        pieces: list[str] = []
        if stream:
            pending: list[int] = []
            for token_id in stream_generate(
                loaded.model,
                torch.tensor([prompt_ids], dtype=torch.long),
                config,
                device=loaded.device,
            ):
                pending.append(token_id)
                text = loaded.tokenizer.decode(pending)
                if "\ufffd" in text:
                    continue
                pieces.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
                pending = []
            print()
        else:
            output = generate(
                loaded.model,
                torch.tensor([prompt_ids], dtype=torch.long),
                config,
                device=loaded.device,
            )
            text = loaded.tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
            pieces.append(text)
            print(text)

        reply = "".join(pieces).strip()
        history.append({"role": "assistant", "content": reply})
        elapsed = time.perf_counter() - started
        n_tokens = len(loaded.tokenizer.encode(reply))
        if elapsed > 0 and n_tokens:
            print(f"  [{n_tokens} tokens, {n_tokens / elapsed:.1f} tok/s]\n")
