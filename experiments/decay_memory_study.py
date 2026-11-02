"""Fairness helpers for the real-data decayed Mini-Batch study.

The existing decayed CUDA methods are deliberately unchanged.  Their batch
factor is observation-scaled, so historical sufficient-statistic retention is
invariant to how a fixed observation budget is partitioned into batches.
"""

from __future__ import annotations

import math
from numbers import Integral, Real


def decay_factor(batch_size: int, n: int, half_life_passes: float = 10.0) -> float:
    """Return the existing block-update factor for one batch."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(half_life_passes, bool) or not isinstance(half_life_passes, Real):
        raise TypeError("half_life_passes must be real")
    if not math.isfinite(half_life_passes) or half_life_passes <= 0:
        raise ValueError("half_life_passes must be finite and positive")
    return 0.5 ** (int(batch_size) / (float(half_life_passes) * int(n)))


def schedule_retention(batch_sizes, n: int, half_life_passes: float = 10.0) -> float:
    """Retention of historical sufficient statistics over a batch schedule."""
    batches = tuple(batch_sizes)
    return math.prod(decay_factor(b, n, half_life_passes) for b in batches)


def expected_retention(observations: int, n: int, half_life_passes: float = 10.0) -> float:
    """Equivalent retention expressed directly in processed observations."""
    if isinstance(observations, bool) or not isinstance(observations, Integral) or observations < 0:
        raise ValueError("observations must be a nonnegative integer")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(half_life_passes, bool) or not isinstance(half_life_passes, Real):
        raise TypeError("half_life_passes must be real")
    if not math.isfinite(half_life_passes) or half_life_passes <= 0:
        raise ValueError("half_life_passes must be finite and positive")
    return 0.5 ** (int(observations) / (float(half_life_passes) * int(n)))


def within_batch_oldest_to_newest_weight_ratio(batch_size: int, n: int,
                                               half_life_passes: float = 10.0) -> float:
    """Exact per-observation weighting ratio omitted by the block approximation."""
    factor = decay_factor(1, n, half_life_passes)
    return factor ** (int(batch_size) - 1)
