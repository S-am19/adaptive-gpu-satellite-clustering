"""Correctness and lifecycle tests for Mini-Batch K-Means with ANN backend."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from pcap.assignments_ann import HNSWCPUBackend
from pcap.contracts import FitConfig

from pcap.assignments_exact import assign_exact
from pcap.minibatch import fit_minibatch_cpu
from pcap.minibatch_ann import fit_minibatch_ann_cpu


@pytest.fixture
def fake_hnsw(monkeypatch):
    """Exact stub for HNSW lifecycle, index rebuilds and accounting tests."""
    instances = []

    class Index:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.batches = []
            instances.append(self)

        def init_index(self, **kwargs):
            self.settings = kwargs

        def set_num_threads(self, n):
            assert n == 1

        def add_items(self, data, ids, num_threads):
            assert num_threads == 1
            np.testing.assert_array_equal(ids, np.arange(len(data)))
            self.data = data.copy()

        def set_ef(self, ef):
            self.ef = ef

        def knn_query(self, data, k, num_threads):
            assert k == 1 and num_threads == 1
            self.batches.append(len(data))
            result = assign_exact(data, self.data)
            return result.labels[:, None], result.squared_distances[:, None]

    monkeypatch.setitem(sys.modules, "hnswlib", SimpleNamespace(Index=Index))
    return instances


def test_ann_minibatch_matches_exact_when_ann_is_exact(fake_hnsw):
    """When the ANN backend behaves exactly, results must match fit_minibatch_cpu."""
    rng = np.random.default_rng(101)
    points = rng.normal(size=(64, 4)).astype(np.float32)
    initial = points[:3].copy()
    cfg_exact = FitConfig(k=3, seed=7, max_steps=12, tolerance=1e-5,
                          batch_size=16, probe_interval=4, ann_effort=None)
    cfg_ann = FitConfig(k=3, seed=7, max_steps=12, tolerance=1e-5,
                        batch_size=16, probe_interval=4, ann_effort=16)

    exact_run = fit_minibatch_cpu(points, initial, cfg_exact)
    ann_run = fit_minibatch_ann_cpu(points, initial, cfg_ann)

    # Identical numerical outputs
    np.testing.assert_allclose(ann_run.centroids, exact_run.centroids, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(ann_run.labels, exact_run.labels)
    np.testing.assert_allclose(ann_run.sse, exact_run.sse, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(
        ann_run.metadata["cluster_observation_counts"],
        exact_run.metadata["cluster_observation_counts"],
    )
    assert ann_run.metadata["steps"] == exact_run.metadata["steps"]
    assert ann_run.metadata["termination_reason"] == exact_run.metadata["termination_reason"]


def test_ann_minibatch_lifecycle_and_accounting(fake_hnsw):
    """Verify index preparation, per-step refreshes, and timing accounting."""
    rng = np.random.default_rng(42)
    points = rng.normal(size=(50, 2)).astype(np.float32)
    initial = points[:2].copy()
    cfg = FitConfig(k=2, seed=0, max_steps=5, tolerance=0.0,
                    batch_size=10, probe_interval=2, ann_effort=8)

    backend = HNSWCPUBackend(seed=0, ann_effort=8)
    result = fit_minibatch_ann_cpu(points, initial, cfg, backend=backend)

    # Check that initial build and rebuilds occurred
    assert backend.timings["initial_build_count"] == 1
    # 5 steps = 5 refreshes
    assert backend.timings["rebuild_count"] == 5
    assert result.metadata["rebuild_count"] == 5

    # Check timings ledger
    timings = result.timings
    assert all(np.isfinite(v) and v >= 0 for v in timings.values())
    assert timings["index_maintenance_s"] > 0
    assert timings["final_assignment_s"] <= timings["compute_s"] <= timings["wall_s"]

    # Probe recall should be 1.0 with exact stub
    for entry in result.history:
        if entry.get("convergence_checked"):
            assert entry["probe_recall"] == 1.0


def test_ann_minibatch_invalid_inputs(fake_hnsw):
    points = np.zeros((10, 2), dtype=np.float32)
    initial = np.zeros((2, 2), dtype=np.float32)

    # Missing ann_effort
    cfg_no_effort = FitConfig(k=2, seed=0, max_steps=5, tolerance=1e-4, batch_size=5)
    with pytest.raises(ValueError, match="ann_effort"):
        fit_minibatch_ann_cpu(points, initial, cfg_no_effort)

    # Missing batch_size
    cfg_no_batch = FitConfig(k=2, seed=0, max_steps=5, tolerance=1e-4, ann_effort=16)
    with pytest.raises(ValueError, match="batch_size"):
        fit_minibatch_ann_cpu(points, initial, cfg_no_batch)

    # Batch size > N
    cfg_huge_batch = FitConfig(k=2, seed=0, max_steps=5, tolerance=1e-4, batch_size=20, ann_effort=16)
    with pytest.raises(ValueError, match="batch_size"):
        fit_minibatch_ann_cpu(points, initial, cfg_huge_batch)


def test_ann_minibatch_empty_clusters_retained(fake_hnsw):
    points = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]], dtype=np.float32)
    initial = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float32)
    cfg = FitConfig(k=2, seed=0, max_steps=3, tolerance=0.0,
                    batch_size=2, probe_interval=5, ann_effort=8)
    result = fit_minibatch_ann_cpu(points, initial, cfg)
    np.testing.assert_array_equal(result.centroids[1], initial[1])
    assert result.metadata["cluster_observation_counts"][1] == 0
