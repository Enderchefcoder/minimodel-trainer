"""Reinforcement learning from verifiable rewards.

RLVR replaces the learned reward model of RLHF with a program that checks the
answer. If the task is "what is 12 + 30", the reward is whether the model said
42. Nothing to train, nothing to hack, no reward drift.

The optimizer is GRPO (Group Relative Policy Optimization): sample ``G``
completions per prompt, and use the group's own mean reward as the baseline
instead of training a value network. That removes half the memory of PPO and
happens to suit small models well, since a value head at 30M parameters is
noisy enough to hurt more than it helps.

The advantage for completion ``i`` of a group is::

    A_i = (r_i - mean(r)) / (std(r) + eps)

which is combined with a clipped importance ratio and a KL penalty toward the
reference policy - the same three ingredients as PPO, minus the critic.

Verifiers
---------
Reward functions live in :data:`VERIFIERS` and are selected by name in the task
file. Registering a new one is a decorator away.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minimodel.core.logging_utils import get_logger
from minimodel.core.registry import Registry
from minimodel.datasets.shards import IGNORE_INDEX
from minimodel.inference.sampling import SamplingConfig, generate
from minimodel.training.trainer import Trainer, TrainerConfig

__all__ = [
    "VERIFIERS",
    "RLVRConfig",
    "RLVRTrainer",
    "extract_final_number",
    "group_advantages",
]

logger = get_logger(__name__)

#: Registry of ``(completion, reference) -> reward in [0, 1]`` functions.
VERIFIERS: Registry[Callable[[str, str], float]] = Registry("verifier")

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def extract_final_number(text: str) -> str | None:
    """Return the last number in ``text``, or ``None``.

    Models trained on GSM8K-style data put the answer last, so the final number
    is a far more reliable target than the first.

    >>> extract_final_number("we get 12 then 30, so 42.")
    '42'
    """
    matches = _NUMBER_PATTERN.findall(text.replace(",", ""))
    return matches[-1] if matches else None


@VERIFIERS.register("numeric", aliases=("number", "gsm8k"))
def verify_numeric(completion: str, reference: str) -> float:
    """1.0 when the last number in the completion equals the reference number."""
    predicted = extract_final_number(completion)
    expected = extract_final_number(reference) or reference.strip()
    if predicted is None:
        return 0.0
    try:
        return 1.0 if math.isclose(float(predicted), float(expected), rel_tol=1e-6) else 0.0
    except ValueError:
        return 1.0 if predicted.strip() == expected.strip() else 0.0


@VERIFIERS.register("exact_match", aliases=("exact", "string"))
def verify_exact_match(completion: str, reference: str) -> float:
    """1.0 when the normalised completion contains the reference exactly."""
    return 1.0 if reference.strip().lower() in completion.strip().lower() else 0.0


@VERIFIERS.register("latex_answer", aliases=("boxed", "math"))
def verify_latex_answer(completion: str, reference: str) -> float:
    """Compare the contents of the last ``\\boxed{...}`` in each string."""
    predicted = _BOXED_PATTERN.findall(completion)
    expected = _BOXED_PATTERN.findall(reference) or [reference]
    if not predicted:
        return verify_numeric(completion, expected[-1])
    return 1.0 if predicted[-1].strip() == expected[-1].strip() else 0.0


@VERIFIERS.register("expression")
def verify_expression(completion: str, reference: str) -> float:
    """Evaluate an arithmetic expression in the completion and compare it.

    Only digits, operators, parentheses and whitespace are permitted, so the
    expression can be evaluated without exposing anything else.
    """
    candidates = re.findall(r"[-+*/() \d.]{3,}", completion)
    try:
        target = float(reference)
    except ValueError:
        return 0.0
    for candidate in reversed(candidates):
        expression = candidate.strip()
        if not expression or not re.fullmatch(r"[-+*/() \d.]+", expression):
            continue
        try:
            value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        except (SyntaxError, ZeroDivisionError, TypeError, NameError, ValueError):
            continue
        if isinstance(value, (int, float)) and math.isclose(value, target, rel_tol=1e-6):
            return 1.0
    return 0.0


@VERIFIERS.register("length")
def verify_length(completion: str, reference: str) -> float:
    """Reward staying close to a target length; useful as a shaping term."""
    try:
        target = float(reference)
    except ValueError:
        return 0.0
    actual = len(completion.split())
    if target <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(actual - target) / target)


def group_advantages(rewards: Tensor, *, eps: float = 1e-4) -> Tensor:
    """Normalise rewards within each group.

    ``rewards`` is ``[n_prompts, group_size]``. Groups where every completion
    scored the same produce zero advantage, which is correct: they carry no
    learning signal.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


@dataclass
class RLVRConfig(TrainerConfig):
    """Configuration for :class:`RLVRTrainer`."""

    run_name: str = "rlvr"
    lr: float = 1e-6
    max_steps: int = 200
    batch_size: int = 4
    seq_len: int = 512
    warmup: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 0.2

    #: Completions sampled per prompt. Larger groups give a lower-variance
    #: baseline; 8 is a reasonable floor, 16 is better if the budget allows.
    group_size: int = 8
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_p: float = 0.95
    #: PPO-style clipping range on the importance ratio.
    clip_epsilon: float = 0.2
    #: Weight of the KL penalty toward the frozen reference policy.
    kl_coefficient: float = 0.04
    #: Default verifier when a task record does not name one.
    verifier: str = "numeric"
    #: Extra prompt appended to every task, e.g. a format instruction.
    system_prompt: str | None = None
    #: Small bonus for producing a well-formed answer at all, which gives the
    #: model gradient before it ever gets a task right.
    format_bonus: float = 0.0
    reward_kwargs: dict[str, Any] = field(default_factory=dict)


class RLVRTrainer(Trainer):
    """GRPO training against programmatic reward functions.

    Parameters
    ----------
    tasks_path:
        JSONL file whose records have ``prompt`` and ``answer`` fields, and
        optionally ``verifier``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RLVRConfig | None = None,
        *,
        tasks_path: str | Path | Sequence[Mapping[str, Any]],
        tokenizer: Any,
        reference_model: nn.Module | None = None,
        **kwargs: Any,
    ):
        config = config or RLVRConfig()
        super().__init__(model, config, tokenizer=tokenizer, **kwargs)
        self.rlvr_config = config
        self.tasks = self._load_tasks(tasks_path)
        if not self.tasks:
            raise ValueError("RLVR needs at least one task")

        self.reference = reference_model or copy.deepcopy(self.raw_model)
        self.reference.to(self.device).eval()
        for param in self.reference.parameters():
            param.requires_grad_(False)

        self._task_cursor = 0
        logger.info("RLVR over %d tasks, group size %d", len(self.tasks), config.group_size)

    @staticmethod
    def _load_tasks(source: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Read the task list from a JSONL path or an in-memory sequence."""
        if not isinstance(source, (str, Path)):
            return [dict(item) for item in source]
        path = Path(source)
        if path.is_dir():
            path = path / "tasks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"RLVR task file not found: {path}")
        tasks: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def _next_tasks(self, count: int) -> list[dict[str, Any]]:
        """Take the next ``count`` tasks, wrapping around."""
        selected = []
        for _ in range(count):
            selected.append(self.tasks[self._task_cursor % len(self.tasks)])
            self._task_cursor += 1
        return selected

    def score(self, completion: str, task: Mapping[str, Any]) -> float:
        """Reward a single completion against its task."""
        verifier_name = str(task.get("verifier") or self.rlvr_config.verifier)
        verifier = VERIFIERS.get(verifier_name)
        reference = str(task.get("answer", ""))
        reward = float(verifier(completion, reference))
        if self.rlvr_config.format_bonus and completion.strip():
            reward += self.rlvr_config.format_bonus
        return reward

    def _build_prompt(self, task: Mapping[str, Any]) -> str:
        prompt = str(task.get("prompt", ""))
        if self.rlvr_config.system_prompt:
            return f"{self.rlvr_config.system_prompt}\n{prompt}"
        return prompt

    @torch.no_grad()
    def rollout(self) -> dict[str, Any]:
        """Sample a group of completions per prompt and score them."""
        config = self.rlvr_config
        tasks = self._next_tasks(config.batch_size)
        sampling = SamplingConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            stop_token_ids=[self.tokenizer.eos_id],
        )

        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        rewards: list[float] = []
        was_training = self.model.training
        self.model.eval()

        for task in tasks:
            prompt_ids = self.tokenizer.encode(self._build_prompt(task), add_bos=True)
            prompt_tensor = torch.tensor([prompt_ids] * config.group_size, dtype=torch.long)
            generated = generate(self.raw_model, prompt_tensor, sampling, device=self.device)
            for row in generated.tolist():
                completion = self.tokenizer.decode(row[len(prompt_ids) :])
                sequences.append(row)
                prompt_lengths.append(len(prompt_ids))
                rewards.append(self.score(completion, task))

        self.model.train(was_training)

        width = max(len(seq) for seq in sequences)
        pad_id = self.tokenizer.pad_id
        input_ids = torch.tensor(
            [seq + [pad_id] * (width - len(seq)) for seq in sequences], dtype=torch.long
        ).to(self.device)

        # Only completion tokens are trained on; the prompt is context.
        labels = input_ids.clone()
        for row, (sequence, prompt_length) in enumerate(
            zip(sequences, prompt_lengths, strict=True)
        ):
            labels[row, :prompt_length] = IGNORE_INDEX
            labels[row, len(sequence) :] = IGNORE_INDEX

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        grouped = reward_tensor.view(len(tasks), config.group_size)
        advantages = group_advantages(grouped).reshape(-1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "advantages": advantages,
            "rewards": reward_tensor,
        }

    def _token_logprobs(self, model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Per-token log-probabilities of ``labels``, zero where masked."""
        logits = model(input_ids)[:, :-1]
        targets = labels[:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        mask = targets != IGNORE_INDEX
        safe = targets.masked_fill(~mask, 0)
        gathered = logprobs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        return gathered * mask.float()

    def _next_batch(self) -> dict[str, Tensor]:
        """A "batch" for RLVR is a freshly sampled and scored rollout."""
        return self.rollout()

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Clipped policy-gradient loss with a KL penalty toward the reference."""
        config = self.rlvr_config
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        advantages = batch["advantages"]

        policy_logprobs = self._token_logprobs(self.raw_model, input_ids, labels)
        with torch.no_grad():
            reference_logprobs = self._token_logprobs(self.reference, input_ids, labels)
            old_logprobs = policy_logprobs.detach()

        mask = (labels[:, 1:] != IGNORE_INDEX).float()
        token_count = mask.sum().clamp(min=1.0)

        ratio = torch.exp(policy_logprobs - old_logprobs)
        advantage_per_token = advantages.unsqueeze(-1)
        unclipped = ratio * advantage_per_token
        clipped = (
            torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            * advantage_per_token
        )
        policy_loss = -(torch.min(unclipped, clipped) * mask).sum() / token_count

        # k3 estimator: unbiased, always non-negative, much lower variance than
        # the naive difference of log-probabilities.
        log_ratio = reference_logprobs - policy_logprobs
        kl = (torch.exp(log_ratio) - log_ratio - 1.0) * mask
        kl_loss = kl.sum() / token_count

        loss = policy_loss + config.kl_coefficient * kl_loss

        rewards = batch["rewards"]
        extras = {
            "reward_mean": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "solve_rate": float((rewards >= 1.0).float().mean()),
            "kl": float(kl_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "completion_tokens": float(token_count / max(1, input_ids.shape[0])),
        }
        return loss, extras

    @torch.no_grad()
    def evaluate(self, loader=None, max_batches: int | None = None) -> dict[str, float]:
        """Greedy-decode the task set and report the solve rate."""
        was_training = self.model.training
        self.model.eval()
        sampling = SamplingConfig(
            max_new_tokens=self.rlvr_config.max_new_tokens,
            do_sample=False,
            stop_token_ids=[self.tokenizer.eos_id],
        )
        limit = min(len(self.tasks), (max_batches or self.config.eval_batches) * 4)
        solved = 0.0
        for task in self.tasks[:limit]:
            prompt_ids = self.tokenizer.encode(self._build_prompt(task), add_bos=True)
            output = generate(
                self.raw_model,
                torch.tensor([prompt_ids], dtype=torch.long),
                sampling,
                device=self.device,
            )
            completion = self.tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
            solved += self.score(completion, task)
        self.model.train(was_training)
        if limit == 0:
            return {}
        return {"val_solve_rate": solved / limit, "val_loss": 1.0 - solved / limit}
