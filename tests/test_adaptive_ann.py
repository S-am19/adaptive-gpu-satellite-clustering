"""Unit tests for adaptive ANN search effort policy and combined adaptive fitting."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from pcap.adaptive_ann_controller import (
    ANNDecision,
    ANNEffortPolicy,
    ANNEffortState,
    initial_ann_state,
    transition_ann,
)
try:
    from pcap.adaptive_combined import fit_adaptive_minibatch
    from pcap.assignments_exact import assign_exact
except ImportError:
    fit_adaptive_minibatch = None
    assign_exact = None
from pcap.adaptive_controller import Checkpoint
from pcap.contracts import FitConfig


@pytest.fixture
def fake_hnsw(monkeypatch):
    """Exact stub for HNSW lifecycle and accounting tests."""
    class Index:
        def __init__(self, **kwargs):
            self.options = kwargs

        def init_index(self, **kwargs):
            self.settings = kwargs

        def set_num_threads(self, n):
            assert n == 1

        def add_items(self, data, ids, num_threads):
            assert num_threads == 1
            self.data = data.copy()

        def set_ef(self, ef):
            self.ef = ef

        def knn_query(self, data, k, num_threads):
            assert k == 1 and num_threads == 1
            result = assign_exact(data, self.data)
            return result.labels[:, None], result.squared_distances[:, None]

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=Index))


def test_ann_policy_validation():
    # Non-descending levels
    with pytest.raises(ValueError, match="strictly descending"):
        ANNEffortPolicy(levels=(8, 16, 32))

    # Duplicate levels
    with pytest.raises(ValueError, match="distinct"):
        ANNEffortPolicy(levels=(64, 32, 32, 16))

    # Initial effort not in levels
    with pytest.raises(ValueError, match="initial_effort"):
        ANNEffortPolicy(levels=(64, 32, 16), initial_effort=8)


def test_ann_transition_quiet_stepdown():
    policy = ANNEffortPolicy(levels=(64, 32, 16, 8), initial_effort=64, quiet_checkpoints=2)
    state = initial_ann_state(policy, 10.0)

    # Checkpoint 1: quiet, but only 1 check
    cp1 = Checkpoint(objective=9.995, movement=0.0005, label_change=0.0)
    state, d1 = transition_ann(policy, state, cp1)
    assert state.effort == 64
    assert d1.action == "retain"
    assert state.quiet_count == 1

    # Checkpoint 2: quiet, reaches 2 quiet checks -> step down to 32
    cp2 = Checkpoint(objective=9.990, movement=0.0005, label_change=0.0)
    state, d2 = transition_ann(policy, state, cp2)
    assert state.effort == 32
    assert d2.action == "decrease"
    assert d2.next_effort == 32
    assert state.quiet_count == 0  # reset on change


def test_ann_transition_rollback_on_deterioration():
    policy = ANNEffortPolicy(levels=(64, 32, 16, 8), initial_effort=64, quiet_checkpoints=1, deterioration=0.01)
    state = initial_ann_state(policy, 10.0)

    # Step down to 32
    state, _ = transition_ann(policy, state, Checkpoint(10.0, 0.0001, 0.0))
    assert state.effort == 32
    assert state.safe_effort == 64

    # Deterioration: objective jumps from 10.0 to 10.2 (>1% worse than safe_objective 10.0)
    state, d = transition_ann(policy, state, Checkpoint(10.2, 0.01, 0.05))
    assert d.action == "rollback"
    assert d.reason == "deterioration"
    assert state.effort == 64  # restored safe effort


@pytest.mark.skipif(fit_adaptive_minibatch is None, reason="numba required for integration fit")
def test_combined_adaptive_modes(fake_hnsw):
    rng = np.random.default_rng(77)
    points = rng.normal(size=(60, 4)).astype(np.float32)
    initial = points[:3].copy()
    cfg = FitConfig(k=3, seed=1, max_steps=10, tolerance=1e-4,
                    batch_size=15, probe_interval=3, ann_effort=64)

    # Mode 1: batch_only (G3)
    res_batch = fit_adaptive_minibatch(points, initial, cfg, mode="batch_only")
    assert res_batch.metadata["mode"] == "batch_only"
    assert res_batch.sse > 0
    assert len(res_batch.labels) == 60

    # Mode 2: ann_only (G4)
    res_ann = fit_adaptive_minibatch(points, initial, cfg, mode="ann_only")
    assert res_ann.metadata["mode"] == "ann_only"
    assert res_ann.sse > 0
    assert res_ann.metadata["ann_effort"] == 64

    # Mode 3: combined (G5)
    res_comb = fit_adaptive_minibatch(points, initial, cfg, mode="combined")
    assert res_comb.metadata["mode"] == "combined"
    assert res_comb.sse > 0
    # Every run must populate final exact SSE
    assert res_comb.sse == pytest.approx(res_comb.metadata["final_exact_sse"])
