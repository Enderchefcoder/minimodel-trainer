"""A tiny learned "is this real text?" probe for reranking under search.

A model grading its own continuations by log-probability Goodharts under heavy
search: it drifts toward whatever it finds most probable, which for a
small model is memorised boilerplate (repeated phrases, section-header markup).
The fix, from Glint-2, is a probe that is *not* self-referential — one linear
layer over the model's mean-pooled hidden state, trained to separate real corpus
text from the model's own generations. Blended into the rerank score, it stops
confident garbage from winning.

The probe is ~`(dim + 2) * 4` bytes (a few KB) and trains in seconds on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from minimodel.core.logging_utils import get_logger

__all__ = ["QualityProbe", "train_quality_probe"]

logger = get_logger(__name__)


class QualityProbe:
    """Linear P(real) head over a model's mean-pooled final hidden state.

    Stores a standardisation (mean, std) plus a linear layer. Kept as plain
    tensors — not an ``nn.Module`` on the model — so it is trivially portable
    and serialisable, matching Glint-2's 3.5 KB artifact.
    """

    def __init__(self, weight: Tensor, bias: Tensor, mean: Tensor, std: Tensor):
        self.weight = weight.flatten().float()
        self.bias = float(bias.flatten()[0])
        self.mean = mean.flatten().float()
        self.std = std.flatten().float().clamp(min=1e-6)

    @property
    def dim(self) -> int:
        return self.weight.numel()

    @torch.no_grad()
    def p_real(
        self, model: nn.Module, tokens: list[int], prompt_len: int, model_kwargs: dict | None = None
    ) -> float:
        """P(the continuation after `prompt_len` is real text)."""
        model_kwargs = model_kwargs or {}
        hidden = model(torch.tensor([tokens]), return_hidden=True, **model_kwargs)
        feat = hidden[0, prompt_len:].mean(dim=0).float()
        z = (feat - self.mean) / self.std
        return float(torch.sigmoid(z @ self.weight + self.bias))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weight": self.weight,
                "bias": torch.tensor([self.bias]),
                "mean": self.mean,
                "std": self.std,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> QualityProbe:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state["weight"], state["bias"], state["mean"], state["std"])

    def __repr__(self) -> str:
        return f"QualityProbe(dim={self.dim})"


@torch.no_grad()
def _pool_hidden(model: nn.Module, ids: list[int], model_kwargs: dict) -> Tensor:
    """Mean-pooled final hidden state for a token sequence."""
    hidden = model(torch.tensor([ids[:512]]), return_hidden=True, **model_kwargs)
    return hidden[0].mean(dim=0).float()


def train_quality_probe(
    model: nn.Module,
    tokenizer: Any,
    real_texts: Sequence[str],
    *,
    n_prompts: int = 64,
    max_new_tokens: int = 64,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
    model_kwargs: dict | None = None,
) -> QualityProbe:
    """Fit a quality probe: real corpus text (positive) vs model samples (negative).

    Prompts for the negatives are the opening tokens of the real texts, so the
    probe learns to tell a real continuation from the model's continuation *of
    the same prompt* — exactly the decision it makes during reranking.
    """
    model_kwargs = model_kwargs or {}
    from minimodel.inference.sampling import SamplingConfig, generate

    was_training = model.training
    model.eval()
    torch.manual_seed(seed)

    real_feats: list[Tensor] = []
    fake_feats: list[Tensor] = []
    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        stop_token_ids=[getattr(tokenizer, "eos_id", 0)],
        model_kwargs=dict(model_kwargs),
    )

    for text in real_texts[:n_prompts]:
        ids = _encode(tokenizer, text)
        if len(ids) < 8:
            continue
        real_feats.append(_pool_hidden(model, ids, model_kwargs))
        prompt_ids = ids[: max(4, len(ids) // 4)]
        out = generate(model, torch.tensor([prompt_ids]), sampling)
        fake_feats.append(_pool_hidden(model, out[0].tolist(), model_kwargs))

    model.train(was_training)
    if len(real_feats) < 4:
        raise ValueError("not enough usable real texts to train a quality probe")

    x = torch.stack(real_feats + fake_feats)
    y = torch.cat([torch.ones(len(real_feats)), torch.zeros(len(fake_feats))])
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    xz = (x - mean) / std

    linear = nn.Linear(xz.shape[1], 1)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(linear(xz).squeeze(-1), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = float(((torch.sigmoid(linear(xz).squeeze(-1)) > 0.5) == (y > 0.5)).float().mean())
    logger.info(
        "trained quality probe: %d real / %d fake, train accuracy %.2f",
        len(real_feats),
        len(fake_feats),
        acc,
    )
    return QualityProbe(linear.weight.detach(), linear.bias.detach(), mean, std)


def _encode(tokenizer: Any, text: str) -> list[int]:
    import inspect

    try:
        if "add_bos" in inspect.signature(tokenizer.encode).parameters:
            return list(tokenizer.encode(text, add_bos=False))
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return list(tokenizer.encode(text))
