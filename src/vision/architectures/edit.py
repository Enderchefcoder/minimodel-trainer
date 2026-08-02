"""Instruction-based image editing.

The model takes a source image and a text instruction ("make it night", "add a
hat") and produces an edited image. Architecturally it is a conditional
diffusion model where the source image is concatenated to the noisy target along
the channel axis, and the instruction is encoded by the same small text encoder
used elsewhere in this package - the InstructPix2Pix formulation.

Channel concatenation rather than cross-attention is deliberate. The output is
spatially aligned with the input for almost every edit, so giving the network
the source at the same pixel coordinates lets it learn "copy unless told
otherwise", which is most of the task. Cross-attention would force it to
reconstruct that alignment from scratch.

Dual guidance
-------------
Editing needs two guidance scales, not one, because there are two conditions
that can be dropped independently:

* ``image_guidance`` controls faithfulness to the source. Too low and the edit
  ignores the input; too high and nothing changes.
* ``text_guidance`` controls how strongly the instruction is followed.

:meth:`ImageEditModel.forward_with_guidance` evaluates all three branches
(full, image-only, unconditional) in one batched pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from minimodel.vision.architectures.dit import DiT, DiTConfig

__all__ = ["ImageEditModel", "ImageEditConfig"]

#: Defaults for the edit model. ``extra_in_channels`` mirrors ``in_channels``
#: so the source image can be concatenated.
ImageEditConfig: dict[str, Any] = {
    **DiTConfig,
    "image_size": 64,
    "patch_size": 4,
    "dim": 512,
    "depth": 12,
    "n_heads": 8,
    "condition": "text",
    "text_vocab_size": 4096,
    "text_max_len": 32,
    "extra_in_channels": 3,
}


class ImageEditModel(DiT):
    """Diffusion transformer conditioned on a source image and an instruction.

    Examples
    --------
    >>> model = ImageEditModel({"image_size": 8, "patch_size": 2, "dim": 32, "depth": 2,
    ...                         "n_heads": 2, "text_vocab_size": 64})
    >>> source = torch.randn(2, 3, 8, 8)
    >>> noisy = torch.randn(2, 3, 8, 8)
    >>> tokens = torch.randint(0, 64, (2, 8))
    >>> tuple(model(noisy, torch.rand(2), reference=source, text_tokens=tokens).shape)
    (2, 3, 8, 8)
    """

    architecture_name = "image-edit"

    def __init__(self, config: Mapping[str, Any] | None = None):
        merged = {**ImageEditConfig, **dict(config or {})}
        merged.setdefault("extra_in_channels", merged["in_channels"])
        super().__init__(merged)

    def forward_with_guidance(
        self,
        x: Tensor,
        t: Tensor,
        *,
        labels: Tensor | None = None,
        text_tokens: Tensor | None = None,
        reference: Tensor | None = None,
        guidance_scale: float = 1.0,
        image_guidance: float | None = None,
    ) -> Tensor:
        """Apply dual (text and image) classifier-free guidance.

        ``guidance_scale`` is the text scale. When ``image_guidance`` is
        ``None`` this reduces to ordinary single-condition guidance.

        The composed prediction is::

            uncond
            + image_guidance * (image_only - uncond)
            + text_guidance  * (full - image_only)
        """
        if image_guidance is None or reference is None:
            return super().forward_with_guidance(
                x,
                t,
                labels=labels,
                text_tokens=text_tokens,
                reference=reference,
                guidance_scale=guidance_scale,
            )

        batch = x.shape[0]
        zeros_reference = torch.zeros_like(reference)
        null_text = torch.zeros_like(text_tokens) if text_tokens is not None else None

        x_triple = torch.cat([x, x, x], dim=0)
        t_triple = torch.cat([t, t, t], dim=0)
        reference_triple = torch.cat([reference, reference, zeros_reference], dim=0)
        text_triple = (
            torch.cat([text_tokens, null_text, null_text], dim=0)
            if text_tokens is not None
            else None
        )
        labels_triple = torch.cat([labels, labels, labels], dim=0) if labels is not None else None

        prediction = self(
            x_triple,
            t_triple,
            labels=labels_triple,
            text_tokens=text_triple,
            reference=reference_triple,
        )
        full = prediction[:batch]
        image_only = prediction[batch : 2 * batch]
        unconditional = prediction[2 * batch :]
        return (
            unconditional
            + image_guidance * (image_only - unconditional)
            + guidance_scale * (full - image_only)
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ImageEditModel:
        """Build from a config mapping, ignoring bookkeeping keys."""
        payload = {k: v for k, v in dict(config).items() if k in ImageEditConfig}
        return cls(payload)
