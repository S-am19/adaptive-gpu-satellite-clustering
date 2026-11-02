import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np

from experiments.decay_memory_study import (
    decay_factor,
    expected_retention,
    schedule_retention,
    within_batch_oldest_to_newest_weight_ratio,
)


def test_observation_scaled_decay_has_same_retention_across_batch_partitions():
    n = 262144
    schedules = (
        [8192] * 5,
        [16384] * 2 + [8192],
        [32768, 8192],
    )
    total = 40960
    expected = expected_retention(total, n)
    for schedule in schedules:
        assert sum(schedule) == total
        assert schedule_retention(schedule, n) == pytest.approx(expected, rel=2e-15)


def test_per_batch_factor_matches_existing_equation():
    assert decay_factor(65536, 262144) == pytest.approx(0.5 ** (65536 / (10 * 262144)))


def test_block_update_omits_only_small_within_batch_aging_at_study_scale():
    # This is the exact geometric weight ratio between the oldest and newest
    # members under per-observation decay; the existing batch aggregate gives
    # both equal weight. The omission is measured, not silently normalized.
    ratio = within_batch_oldest_to_newest_weight_ratio(65536, 262144)
    assert ratio == pytest.approx(0.5 ** (65535 / (10 * 262144)))
    assert 0.98 < ratio < 1.0


def test_opt_in_cumulative_diagnostic_trace_records_counts_movement_and_sampling():
    from pcap.budgeted_minibatch import fit_budgeted_minibatch
    from pcap.contracts import FitConfig

    X = np.ascontiguousarray(np.array([
        [-2.0, -1.0], [-1.8, -1.1], [-1.5, -0.8],
        [2.0, 1.0], [1.8, 1.2], [1.6, 0.7],
        [-2.1, -0.9], [1.9, 0.9],
    ], dtype=np.float32))
    C0 = np.ascontiguousarray(X[[0, 3]].copy())
    result, checkpoints, _ = fit_budgeted_minibatch(
        X, C0, FitConfig(k=2, seed=7, max_steps=4, tolerance=0.0,
                         batch_size=2, probe_interval=20),
        device="cpu", observation_budget=8, evaluation_interval=4,
        disjoint_probe=False, diagnostic_trace=True,
    )
    assert len(checkpoints) == 2
    assert [sum(row["cluster_update_counts"]) for row in checkpoints] == [4, 8]
    assert all(np.isfinite(row["checkpoint_centroid_movement"]) for row in checkpoints)
    assert all(len(row["per_cluster_centroid_movement"]) == 2 for row in checkpoints)
    assert all(len(row["sample_indices_sha256"]) == 64 for row in result.history)
    assert result.metadata["diagnostic_trace_enabled"] is True


@pytest.mark.parametrize("args", [(0, 10), (1, 0), (-1, 10)])
def test_decay_inputs_reject_invalid_dimensions(args):
    with pytest.raises(ValueError):
        decay_factor(args[0], args[1])
