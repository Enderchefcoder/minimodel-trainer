"""Generation, chat and model loading for inference."""

from __future__ import annotations

from minimodel.inference.run import (
    LoadedModel,
    chat_loop,
    complete,
    complete_batch,
    generate_with_reasoning,
    load_for_inference,
    stream_completion,
)
from minimodel.inference.sampling import (
    SamplingConfig,
    apply_penalties,
    filter_logits,
    generate,
    generate_batch,
    generate_text,
    stream_generate,
)

__all__ = [
    "LoadedModel",
    "SamplingConfig",
    "apply_penalties",
    "chat_loop",
    "complete",
    "complete_batch",
    "filter_logits",
    "generate",
    "generate_batch",
    "generate_text",
    "generate_with_reasoning",
    "load_for_inference",
    "stream_completion",
    "stream_generate",
]
