"""Generation, chat and model loading for inference."""

from __future__ import annotations

from minimodel.inference.quality_probe import QualityProbe, train_quality_probe
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
from minimodel.inference.search import (
    EFFORT_LEVELS,
    EffortConfig,
    effort_generate,
    score_continuation,
)

__all__ = [
    "EFFORT_LEVELS",
    "EffortConfig",
    "LoadedModel",
    "QualityProbe",
    "SamplingConfig",
    "apply_penalties",
    "chat_loop",
    "complete",
    "complete_batch",
    "effort_generate",
    "filter_logits",
    "generate",
    "generate_batch",
    "generate_text",
    "generate_with_reasoning",
    "load_for_inference",
    "score_continuation",
    "stream_completion",
    "stream_generate",
    "train_quality_probe",
]
