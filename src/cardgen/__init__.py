"""Automatic model card generation from run artifacts."""

from __future__ import annotations

from minimodel.cardgen.modelcard_autogen import (
    TEMPLATE_DIR,
    ModelCard,
    ModelCardData,
    collect_card_data,
    generate_model_card,
    render_card,
)

__all__ = [
    "TEMPLATE_DIR",
    "ModelCard",
    "ModelCardData",
    "collect_card_data",
    "generate_model_card",
    "render_card",
]
