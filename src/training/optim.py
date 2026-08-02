"""Optimizers and parameter grouping.

Three optimizers are available:

``adamw``
    The safe default. Well understood, works at every scale.
``muon``
    Orthogonalises the momentum update of 2D parameters with a Newton-Schulz
    iteration. On small models it typically reaches a given loss in noticeably
    fewer steps than AdamW at the same wall-clock cost per step. It only applies
    to matrices, so embeddings, norms and biases stay on AdamW - which is why
    :func:`build_optimizer` returns a wrapper holding both.
``lion``
    Sign-based updates with a single momentum buffer: half the optimizer memory
    of AdamW. Useful when optimizer state, not activations, is the constraint.

Weight decay is applied only to matrices. Decaying norm gains, biases and
embeddings has no regularising benefit and measurably hurts small models.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from minimodel.core.logging_utils import get_logger
from minimodel.core.registry import Registry

__all__ = [
    "OPTIMIZERS",
    "CombinedOptimizer",
    "Lion",
    "Muon",
    "build_optimizer",
    "param_groups",
    "zeropower_via_newtonschulz",
]

logger = get_logger(__name__)

#: Registry of optimizer factories keyed by name.
OPTIMIZERS: Registry[Any] = Registry("optimizer")


def param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups.

    Any parameter with 2 or more dimensions is treated as a matrix and decayed;
    everything else (norm gains, biases, gate vectors, per-head logits) is not.

    >>> import torch.nn as nn
    >>> groups = param_groups(nn.Linear(4, 4), weight_decay=0.1)
    >>> [g["weight_decay"] for g in groups]
    [0.1, 0.0]
    """
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def zeropower_via_newtonschulz(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor of ``matrix`` (its "zeroth power").

    Runs a quintic Newton-Schulz iteration on the normalised matrix, which
    drives every singular value toward 1 without ever computing an SVD. The
    coefficients are the ones tuned for fast convergence in the Muon reference
    implementation; they do not converge to machine precision, but the resulting
    update direction is what matters, not its exact orthogonality.

    Runs in bfloat16 because the iteration is numerically forgiving and this is
    on the critical path of every step.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(Optimizer):
    """MomentUm Orthogonalised by Newton-Schulz.

    Only accepts 2D parameters. Pair it with AdamW for everything else, which
    :func:`build_optimizer` does automatically.

    Parameters
    ----------
    lr:
        Learning rate. Muon tolerates - and usually wants - a larger value than
        AdamW for the same model, typically 10-50x.
    momentum:
        Heavy-ball coefficient on the gradient.
    nesterov:
        Use the Nesterov form of the momentum update.
    ns_steps:
        Newton-Schulz iterations per step. 5 is plenty.
    weight_decay:
        Decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        *,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": int(ns_steps),
            "weight_decay": weight_decay,
        }
        super().__init__(list(params), defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.dim() != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters, got one with {param.dim()} dims. "
                        "Use build_optimizer(), which routes non-matrices to AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if group["nesterov"] else buffer
                update = zeropower_via_newtonschulz(update, steps=group["ns_steps"])
                # Scale by the aspect ratio so wide and tall matrices take
                # comparably sized steps.
                scale = max(1.0, param.size(0) / param.size(1)) ** 0.5
                if group["weight_decay"]:
                    param.mul_(1.0 - lr * group["weight_decay"])
                param.add_(update, alpha=-lr * scale)
        return loss


class Lion(Optimizer):
    """EvoLved Sign Momentum.

    The update is the *sign* of an interpolation between the gradient and the
    momentum buffer, so only one state tensor per parameter is needed. Because
    every update has unit magnitude, the learning rate should be roughly 3-10x
    smaller than AdamW's and the weight decay correspondingly larger.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        *,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        super().__init__(list(params), {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                if group["weight_decay"]:
                    param.mul_(1.0 - lr * group["weight_decay"])
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign_()
                param.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class CombinedOptimizer(Optimizer):
    """Presents several optimizers as one.

    Muon needs a partner for non-matrix parameters. Rather than making the
    trainer aware of that, this wrapper forwards ``step``, ``zero_grad`` and
    state serialisation to each child, and exposes a merged ``param_groups`` so
    learning-rate schedulers work unchanged.
    """

    def __init__(self, optimizers: list[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        self.optimizers = list(optimizers)
        self.defaults = dict(self.optimizers[0].defaults)
        # Deliberately does not call Optimizer.__init__: the parameters are
        # already owned by the children, and re-registering them would make
        # every parameter appear twice.
        self.state = {}  # type: ignore[assignment]

    @property
    def param_groups(self) -> list[dict[str, Any]]:  # type: ignore[override]
        """All child parameter groups, in child order."""
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        # Schedulers mutate `group["lr"]` in place rather than reassigning the
        # list, so this setter only needs to exist, not to do anything.
        return

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """Clear gradients on every child."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        """Step every child."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Serialise every child."""
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Restore every child."""
        for optimizer, state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(state)

    def __repr__(self) -> str:
        names = ", ".join(type(o).__name__ for o in self.optimizers)
        return f"CombinedOptimizer({names})"


@OPTIMIZERS.register("adamw", aliases=("adam",))
def _build_adamw(model: nn.Module, **kwargs: Any) -> Optimizer:
    """AdamW over decay/no-decay parameter groups."""
    lr = float(kwargs.get("lr", 3e-4))
    betas = tuple(kwargs.get("betas", (0.9, 0.95)))
    eps = float(kwargs.get("eps", 1e-8))
    weight_decay = float(kwargs.get("weight_decay", 0.1))
    fused = bool(kwargs.get("fused", False)) and torch.cuda.is_available()
    return torch.optim.AdamW(
        param_groups(model, weight_decay),
        lr=lr,
        betas=betas,  # type: ignore[arg-type]
        eps=eps,
        fused=fused or None,
    )


@OPTIMIZERS.register("lion")
def _build_lion(model: nn.Module, **kwargs: Any) -> Optimizer:
    """Lion over decay/no-decay parameter groups."""
    lr = float(kwargs.get("lr", 1e-4))
    betas = tuple(kwargs.get("betas", (0.9, 0.99)))
    weight_decay = float(kwargs.get("weight_decay", 1.0))
    return Lion(  # type: ignore[return-value]
        param_groups(model, weight_decay),
        lr=lr,
        betas=betas,  # type: ignore[arg-type]
    )


@OPTIMIZERS.register("muon", aliases=("muon_adamw",))
def _build_muon(model: nn.Module, **kwargs: Any) -> Optimizer:
    """Muon for matrices, AdamW for everything else.

    Embeddings and the LM head stay on AdamW even though they are 2D: their
    gradients are extremely sparse per step, and orthogonalising a mostly-zero
    matrix is not meaningful.
    """
    lr = float(kwargs.get("lr", 0.02))
    adamw_lr = float(kwargs.get("adamw_lr", kwargs.get("aux_lr", 3e-4)))
    momentum = float(kwargs.get("momentum", 0.95))
    weight_decay = float(kwargs.get("weight_decay", 0.1))
    ns_steps = int(kwargs.get("ns_steps", 5))

    embedding_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embedding_ids.add(id(module.weight))

    matrices: list[Tensor] = []
    others: list[Tensor] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        is_head = "lm_head" in name or (name.endswith("proj.weight") and "embedding" in name)
        if param.dim() == 2 and id(param) not in embedding_ids and not is_head:
            matrices.append(param)
        else:
            others.append(param)

    logger.debug("muon: %d matrices, %d auxiliary tensors", len(matrices), len(others))
    optimizers: list[Optimizer] = []
    if matrices:
        optimizers.append(
            Muon(matrices, lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        )
    if others:
        decay = [p for p in others if p.dim() >= 2]
        no_decay = [p for p in others if p.dim() < 2]
        optimizers.append(
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=adamw_lr,
                betas=tuple(kwargs.get("betas", (0.9, 0.95))),  # type: ignore[arg-type]
            )
        )
    if len(optimizers) == 1:
        return optimizers[0]
    return CombinedOptimizer(optimizers)  # type: ignore[return-value]


@OPTIMIZERS.register("sgd")
def _build_sgd(model: nn.Module, **kwargs: Any) -> Optimizer:
    """Plain SGD with momentum; mostly useful as a baseline."""
    return torch.optim.SGD(
        param_groups(model, float(kwargs.get("weight_decay", 0.0))),
        lr=float(kwargs.get("lr", 0.1)),
        momentum=float(kwargs.get("momentum", 0.9)),
        nesterov=bool(kwargs.get("nesterov", True)),
    )


def build_optimizer(model: nn.Module, name: str = "adamw", **kwargs: Any) -> Optimizer:
    """Build an optimizer by registry name.

    >>> import torch.nn as nn
    >>> opt = build_optimizer(nn.Linear(4, 4), "adamw", lr=1e-3)
    >>> type(opt).__name__
    'AdamW'
    """
    factory = OPTIMIZERS.get(name)
    return factory(model, **kwargs)
