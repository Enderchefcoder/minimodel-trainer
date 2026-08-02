"""ODE and SDE samplers for the diffusion models.

Sampling means integrating backwards from pure noise at ``t=1`` to data at
``t=0``. For a rectified-flow model the trajectory is close to straight, so a
plain Euler integrator with 20-50 steps already produces good samples; the Heun
sampler halves the discretisation error for the cost of a second model
evaluation per step, which is worth it below about 20 steps.

DDPM's ancestral sampler and the deterministic DDIM sampler are provided for
models trained with the epsilon objective.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from minimodel.vision.training.diffusion import cosine_alpha_bar

__all__ = [
    "SAMPLERS",
    "ddim_sample",
    "ddpm_sample",
    "euler_sample",
    "heun_sample",
    "sample_images",
    "timestep_schedule",
]


def timestep_schedule(
    n_steps: int, *, shift: float = 1.0, device: torch.device | None = None
) -> Tensor:
    """Descending timesteps from 1 to 0.

    ``shift > 1`` spends more of the budget at high noise, where the coarse
    structure of the image is decided. That matters more at higher resolutions;
    at 32px a shift of 1 is usually best.
    """
    t = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    if shift != 1.0:
        t = (shift * t) / (1.0 + (shift - 1.0) * t)
    return t


def _model_call(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    *,
    guidance_scale: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
) -> Tensor:
    """Call the model, using guided inference when the model supports it."""
    kwargs = dict(model_kwargs or {})
    guided = getattr(model, "forward_with_guidance", None)
    if guidance_scale != 1.0 and callable(guided):
        return guided(x, t, guidance_scale=guidance_scale, **kwargs)
    return model(x, t, **kwargs)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    shape: Sequence[int],
    *,
    n_steps: int = 50,
    guidance_scale: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    shift: float = 1.0,
    noise: Tensor | None = None,
    callback: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    """First-order Euler integration of the flow-matching ODE."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = (
        noise.to(device)
        if noise is not None
        else torch.randn(*shape, device=device, generator=generator)
    )
    timesteps = timestep_schedule(n_steps, shift=shift, device=device)

    was_training = model.training
    model.eval()
    for index in range(n_steps):
        t_current = timesteps[index]
        t_next = timesteps[index + 1]
        t_batch = t_current.expand(x.shape[0])
        velocity = _model_call(
            model, x, t_batch, guidance_scale=guidance_scale, model_kwargs=model_kwargs
        )
        x = x + (t_next - t_current) * velocity
        if callback is not None:
            callback(index, x)
    model.train(was_training)
    return x


@torch.no_grad()
def heun_sample(
    model: nn.Module,
    shape: Sequence[int],
    *,
    n_steps: int = 25,
    guidance_scale: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    shift: float = 1.0,
    noise: Tensor | None = None,
) -> Tensor:
    """Second-order Heun integration; two model calls per step."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = (
        noise.to(device)
        if noise is not None
        else torch.randn(*shape, device=device, generator=generator)
    )
    timesteps = timestep_schedule(n_steps, shift=shift, device=device)

    was_training = model.training
    model.eval()
    for index in range(n_steps):
        t_current = timesteps[index]
        t_next = timesteps[index + 1]
        dt = t_next - t_current
        velocity = _model_call(
            model, x, t_current.expand(x.shape[0]), guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
        )
        x_euler = x + dt * velocity
        if index < n_steps - 1:
            velocity_next = _model_call(
                model, x_euler, t_next.expand(x.shape[0]), guidance_scale=guidance_scale,
                model_kwargs=model_kwargs,
            )
            x = x + dt * 0.5 * (velocity + velocity_next)
        else:
            x = x_euler
    model.train(was_training)
    return x


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    shape: Sequence[int],
    *,
    n_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    """DDIM sampling for epsilon-prediction models.

    ``eta=0`` is deterministic; ``eta=1`` recovers ancestral DDPM sampling.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = (
        noise.to(device)
        if noise is not None
        else torch.randn(*shape, device=device, generator=generator)
    )
    timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    was_training = model.training
    model.eval()
    for index in range(n_steps):
        t_current = timesteps[index]
        t_next = timesteps[index + 1]
        alpha_bar = cosine_alpha_bar(t_current).view(1, *([1] * (x.dim() - 1)))
        alpha_bar_next = cosine_alpha_bar(t_next).view(1, *([1] * (x.dim() - 1)))

        epsilon = _model_call(
            model, x, t_current.expand(x.shape[0]), guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
        )
        x0 = (x - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()
        x0 = x0.clamp(-1.0, 1.0)

        sigma = (
            eta
            * ((1 - alpha_bar_next) / (1 - alpha_bar)).sqrt()
            * (1 - alpha_bar / alpha_bar_next).sqrt()
        )
        direction = (1 - alpha_bar_next - sigma**2).clamp(min=0).sqrt() * epsilon
        x = alpha_bar_next.sqrt() * x0 + direction
        if eta > 0 and index < n_steps - 1:
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)
    model.train(was_training)
    return x


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    shape: Sequence[int],
    *,
    n_steps: int = 100,
    guidance_scale: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Ancestral DDPM sampling (DDIM with ``eta=1``)."""
    return ddim_sample(
        model,
        shape,
        n_steps=n_steps,
        eta=1.0,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        device=device,
        generator=generator,
    )


#: Name -> sampler function.
SAMPLERS: dict[str, Callable[..., Tensor]] = {
    "euler": euler_sample,
    "heun": heun_sample,
    "ddim": ddim_sample,
    "ddpm": ddpm_sample,
}


def sample_images(
    model: nn.Module,
    *,
    n_samples: int = 4,
    image_size: int | None = None,
    channels: int = 3,
    sampler: str = "euler",
    n_steps: int = 50,
    guidance_scale: float = 1.0,
    labels: Tensor | None = None,
    text_tokens: Tensor | None = None,
    reference: Tensor | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    **sampler_kwargs: Any,
) -> Tensor:
    """Generate a batch of images in ``[-1, 1]``.

    The image size and channel count default to the model's own config, so most
    callers only need to say how many samples they want.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; available: {', '.join(sorted(SAMPLERS))}")
    device = torch.device(device) if device is not None else next(model.parameters()).device

    config = getattr(model, "config", {})
    image_size = int(image_size or config.get("image_size", 32))
    channels = int(config.get("out_channels") or config.get("in_channels") or channels)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    model_kwargs: dict[str, Any] = {}
    if labels is not None:
        model_kwargs["labels"] = labels.to(device)
    if text_tokens is not None:
        model_kwargs["text_tokens"] = text_tokens.to(device)
    if reference is not None:
        model_kwargs["reference"] = reference.to(device)

    return SAMPLERS[sampler](
        model,
        (n_samples, channels, image_size, image_size),
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        device=device,
        generator=generator,
        **sampler_kwargs,
    )
