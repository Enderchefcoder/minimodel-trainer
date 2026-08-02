"""Self-Play fIne-tuNing (SPIN).

SPIN turns a plain SFT dataset into preference data with no human labels and no
reward model. At iteration ``t``, the model from iteration ``t-1`` generates a
response to each prompt; the human-written response becomes "chosen" and the
model's own becomes "rejected". Training on those pairs with a DPO-style loss
pushes the model toward the human distribution and away from its own current
one.

Why this works when a second SFT epoch would not: the negative examples are
exactly the model's current failure modes, so the gradient targets the gap that
still exists rather than re-reinforcing what the model already does. It keeps
helping for a few iterations and then plateaus, once the model's samples become
hard to distinguish from the data.

Practical notes for small models:

* 2-3 iterations is the useful range; beyond that the synthetic negatives get
  too close to the positives and the gradient vanishes.
* Sampling temperature matters. Too low and the negatives are degenerate and
  trivially separable; ``0.8-1.0`` gives negatives that are actually informative.
* Regenerate the negatives at the start of every iteration. Reusing stale
  samples is the most common way to get a SPIN run that does nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.core.io_utils import ensure_dir
from minimodel.core.logging_utils import get_logger
from minimodel.inference.sampling import SamplingConfig, generate
from minimodel.tokenization.chat import ChatTemplate, normalize_messages
from minimodel.training.rl.dpo import DPOConfig, DPOTrainer

__all__ = ["SPINConfig", "SPINTrainer", "generate_self_play_pairs"]

logger = get_logger(__name__)


@dataclass
class SPINConfig(DPOConfig):
    """Configuration for :class:`SPINTrainer`."""

    run_name: str = "spin"
    lr: float = 5e-7
    beta: float = 0.1
    max_steps: int = 300
    batch_size: int = 4

    #: Number of self-play iterations. Each regenerates the negatives.
    iterations: int = 2
    #: Steps per iteration; ``max_steps`` is the total across all iterations.
    steps_per_iteration: int = 0
    #: Sampling temperature used to produce the rejected responses.
    sample_temperature: float = 0.9
    max_new_tokens: int = 128
    #: Prompts sampled per iteration. 0 uses the whole dataset.
    prompts_per_iteration: int = 0
    #: Directory for the generated pair files, for inspection and reuse.
    pair_cache_dir: str = ""


def generate_self_play_pairs(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    template: ChatTemplate | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    max_length: int = 512,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a preference file where the model's own output is the negative.

    Each record must contain a prompt and a human reference answer; the shapes
    accepted are the same as
    :func:`~minimodel.tokenization.chat.normalize_messages`.
    """
    template = template or ChatTemplate(tokenizer)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        stop_token_ids=template.stop_token_ids(),
        seed=seed,
    )

    was_training = model.training
    model.eval()
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            try:
                messages = normalize_messages(record)
            except (ValueError, TypeError):
                skipped += 1
                continue
            reference = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"), None
            )
            if reference is None:
                skipped += 1
                continue
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_ids = template.render_prompt(prompt_messages)

            with torch.no_grad():
                output = generate(
                    model,
                    torch.tensor([prompt_ids], dtype=torch.long),
                    sampling,
                    device=device,
                )
            generated = tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
            if not generated.strip():
                # An empty sample carries no information about what to avoid.
                skipped += 1
                continue

            chosen = template.render(
                [*prompt_messages, {"role": "assistant", "content": str(reference)}]
            )
            rejected = template.render(
                [*prompt_messages, {"role": "assistant", "content": generated}]
            )
            handle.write(
                json.dumps(
                    {
                        "prompt_ids": prompt_ids[:max_length],
                        "chosen_ids": chosen.input_ids[:max_length],
                        "chosen_labels": chosen.labels[:max_length],
                        "rejected_ids": rejected.input_ids[:max_length],
                        "rejected_labels": rejected.labels[:max_length],
                    }
                )
                + "\n"
            )
            written += 1

    model.train(was_training)
    logger.info("generated %d self-play pairs (%d skipped) -> %s", written, skipped, output_path)
    return {"path": str(output_path), "pairs": written, "skipped": skipped}


class SPINTrainer(DPOTrainer):
    """Iterated self-play fine-tuning.

    Parameters
    ----------
    dataset_path:
        JSONL of prompt/answer records (the same SFT data used earlier), or a
        pre-built preference file when ``regenerate=False``.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SPINConfig | None = None,
        *,
        dataset_path: str | Path,
        tokenizer: Any,
        records: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ):
        config = config or SPINConfig()
        self.spin_config = config
        self.source_records = list(records) if records is not None else self._load_records(dataset_path)
        if not self.source_records:
            raise ValueError(f"no usable SPIN records in {dataset_path}")

        self.template = ChatTemplate(tokenizer)
        self.iteration = 0
        cache_dir = Path(config.pair_cache_dir or (Path(config.output_dir) / config.run_name / "pairs"))
        self.pair_cache_dir = ensure_dir(cache_dir)

        # Iteration 0's negatives come from the starting model, so they have to
        # be generated before the DPO machinery is constructed.
        initial = self.pair_cache_dir / "iteration_000.jsonl"
        generate_self_play_pairs(
            model,
            tokenizer,
            self.source_records if not config.prompts_per_iteration
            else self.source_records[: config.prompts_per_iteration],
            initial,
            template=self.template,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sample_temperature,
            max_length=config.max_pair_length,
            seed=config.seed,
        )
        super().__init__(model, config, pairs_path=initial, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def _load_records(path: str | Path) -> list[dict[str, Any]]:
        """Read prompt/answer records from a JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SPIN dataset not found: {path}")
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def regenerate_pairs(self) -> Path:
        """Produce the next iteration's negatives from the current policy."""
        self.iteration += 1
        target = self.pair_cache_dir / f"iteration_{self.iteration:03d}.jsonl"
        subset = (
            self.source_records
            if not self.spin_config.prompts_per_iteration
            else self.source_records[: self.spin_config.prompts_per_iteration]
        )
        generate_self_play_pairs(
            self.raw_model,
            self.tokenizer,
            subset,
            target,
            template=self.template,
            max_new_tokens=self.spin_config.max_new_tokens,
            temperature=self.spin_config.sample_temperature,
            max_length=self.spin_config.max_pair_length,
            device=self.device,
            seed=(self.spin_config.seed or 0) + self.iteration,
        )
        from minimodel.training.rl.dpo import _PairIterator

        self.pairs = _PairIterator(
            target,
            self.spin_config.batch_size,
            self.spin_config.max_pair_length,
            pad_id=self.tokenizer.pad_id,
        )
        self._pair_iter = self.pairs.batches()
        return target

    def _steps_per_iteration(self) -> int:
        if self.spin_config.steps_per_iteration > 0:
            return self.spin_config.steps_per_iteration
        return max(1, self.config.max_steps // max(1, self.spin_config.iterations))

    def compute_loss(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """DPO loss, with the current self-play iteration recorded."""
        loss, extras = super().compute_loss(batch)
        extras["spin_iteration"] = float(self.iteration)
        return loss, extras

    def train_step(self) -> dict[str, Any]:
        """Run one step, regenerating negatives at each iteration boundary."""
        boundary = self._steps_per_iteration()
        if (
            self.step > 0
            and self.step % boundary == 0
            and self.iteration + 1 < self.spin_config.iterations
        ):
            logger.info("SPIN iteration %d complete, regenerating negatives", self.iteration)
            self.regenerate_pairs()
            # The policy at the end of an iteration becomes the next reference,
            # which is what makes each iteration a fresh comparison rather than
            # a drift away from the original model.
            import copy

            self.reference = copy.deepcopy(self.raw_model).eval()
            for param in self.reference.parameters():
                param.requires_grad_(False)
        return super().train_step()
