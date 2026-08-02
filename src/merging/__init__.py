"""Weight-space model merging.

See :mod:`minimodel.merging.slerp` for the methods and ``docs/merging.md`` for
guidance on which to pick.
"""

from __future__ import annotations

from minimodel.merging.slerp import (
    MERGE_METHODS,
    dare_merge,
    linear_merge,
    load_state_dicts,
    merge_models,
    slerp,
    slerp_merge,
    task_arithmetic_merge,
    ties_merge,
)

__all__ = [
    "MERGE_METHODS",
    "dare_merge",
    "linear_merge",
    "load_state_dicts",
    "merge_models",
    "slerp",
    "slerp_merge",
    "task_arithmetic_merge",
    "ties_merge",
]
