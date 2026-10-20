import numpy as np
import pytest

from pcap.assignments_exact import ExactCPUBackend, accumulate_cluster_stats, assign_exact
from pcap.contracts import FitConfig, RunResult
from pcap.initialization import initialize_centroids
from pcap.kmeans_cpu import _lloyd, fit_cpu


def config(k=2, **kwargs):
    return FitConfig(k=k, seed=42, max_steps=kwargs.pop("max_steps", 30),
                     tolerance=kwargs.pop("tolerance", 0.0), **kwargs)


def numpy_oracle(points, centroids):
    # A full distance matrix is acceptable only for these tiny test fixtures.
    distances = ((points.astype(np.float64)[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
    labels = distances.argmin(1)
    return labels, distances[np.arange(len(points)), labels]


def test_known_lloyd_solution(two_groups):
    points, initial = two_groups
    original_points, original_initial = points.copy(), initial.copy()
    result = fit_cpu(points, initial, config())
    np.testing.assert_array_equal(result.centroids, [[0, 1], [10, 11]])
    np.testing.assert_array_equal(result.labels, [0, 0, 1, 1])
    assert result.sse == 4.0
    assert [entry["sse"] for entry in result.history] == [8.0, 4.0, 4.0]
    assert result.metadata["termination_reason"] == "tolerance"
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(initial, original_initial)
    assert not np.shares_memory(result.centroids, initial)
    assert result.centroids.dtype == np.float32
    assert result.labels.dtype == np.int32
    assert _lloyd.nopython_signatures
    assert not _lloyd.targetoptions.get("parallel", False)


def test_assignment_ties_and_precision():
    points = np.array([[1], [3], [-1]], dtype=np.float32)
    centroids = np.array([[0], [2], [2]], dtype=np.float32)
    result = assign_exact(points, centroids)
    np.testing.assert_array_equal(result.labels, [0, 1, 0])
    np.testing.assert_array_equal(result.squared_distances, [1, 1, 1])
    assert result.squared_distances.dtype == np.float64
    # Squaring in float32 would overflow; the reference accumulates in float64.
    large = assign_exact(np.array([[3e38]], np.float32), np.array([[-3e38]], np.float32))
    assert np.isfinite(large.squared_distances).all()


def test_sums_counts_and_empty_cluster(two_groups):
    points, _ = two_groups
    labels = np.array([0, 0, 1, 1], np.int32)
    stats = accumulate_cluster_stats(points, labels, 3)
    np.testing.assert_array_equal(stats.sums, [[0, 2], [20, 22], [0, 0]])
    np.testing.assert_array_equal(stats.counts, [2, 2, 0])
    assert stats.sums.dtype == np.float64
    assert stats.counts.dtype == np.int64


def test_empty_centroid_is_retained():
    points = np.array([[0], [2], [10], [12]], np.float32)
    initial = np.array([[0], [10], [100]], np.float32)
    result = fit_cpu(points, initial, config(k=3))
    np.testing.assert_array_equal(result.centroids, [[1], [11], [100]])
    assert np.isfinite(result.centroids).all()


def test_duplicate_centroids_and_identical_points():
    points = np.ones((4, 2), np.float32)
    result = fit_cpu(points, points[:3].copy(), config(k=3))
    np.testing.assert_array_equal(result.labels, np.zeros(4, np.int32))
    np.testing.assert_array_equal(result.centroids, np.ones((3, 2)))
    assert result.sse == 0


def test_final_assignment_after_step_limit():
    points = np.array([[0], [2], [3], [10]], np.float32)
    initial = np.array([[0], [3]], np.float32)
    before = assign_exact(points, initial).labels
    result = fit_cpu(points, initial, config(max_steps=1))
    expected_labels, expected_distances = numpy_oracle(points, result.centroids)
    assert not np.array_equal(before, expected_labels)
    np.testing.assert_array_equal(result.labels, expected_labels)
    assert result.sse == pytest.approx(expected_distances.sum())
    assert result.metadata["termination_reason"] == "max_steps"


@pytest.mark.parametrize("seed", [0, 1, 42, 1234, 9999])
def test_reproducibility_and_monotonic_objective(four_features, seed):
    points = four_features
    initial = initialize_centroids(points, 3, seed=seed)
    np.testing.assert_array_equal(initial, initialize_centroids(points, 3, seed=seed))
    cfg = FitConfig(3, seed, 40, 1e-6)
    first = fit_cpu(points, initial, cfg)
    second = fit_cpu(points, initial, cfg)
    np.testing.assert_array_equal(first.centroids, second.centroids)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.history == second.history
    objectives = np.array([entry["sse"] for entry in first.history])
    assert np.all(np.diff(objectives) <= 1e-10 * np.maximum(1, objectives[:-1]))
    labels, distances = numpy_oracle(points, first.centroids)
    np.testing.assert_array_equal(first.labels, labels)
    assert first.sse == pytest.approx(distances.sum(), rel=1e-14)


def test_initialization_samples_distinct_rows_without_aliasing():
    points = np.arange(24, dtype=np.float32).reshape(6, 4)
    initial = initialize_centroids(points, 6, seed=0)
    assert len(np.unique(initial, axis=0)) == 6
    assert not np.shares_memory(initial, points)


def test_tolerance_and_single_point():
    points = np.array([[2, 3]], np.float32)
    result = fit_cpu(points, points.copy(), config(k=1))
    assert result.sse == 0
    assert result.metadata["steps"] == 1
    points = np.array([[0], [2]], np.float32)
    result = fit_cpu(points, points[:1].copy(), config(k=1, tolerance=2))
    assert result.metadata["steps"] == 1
    assert result.sse == 2


def test_backend_adapter(two_groups):
    points, initial = two_groups
    backend = ExactCPUBackend()
    backend.prepare(initial)
    backend.refresh(initial)
    np.testing.assert_array_equal(backend.assign(points, initial).labels, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="effort"):
        backend.assign(points, initial, effort=1)


@pytest.mark.parametrize("field,value,error", [
    ("k", 0, ValueError), ("k", True, TypeError), ("k", 2.0, TypeError),
    ("seed", -1, ValueError), ("seed", False, TypeError),
    ("max_steps", 0, ValueError), ("max_steps", 1.5, TypeError),
    ("tolerance", -1, ValueError), ("tolerance", np.nan, ValueError),
    ("tolerance", np.inf, ValueError), ("tolerance", True, TypeError),
    ("tolerance", "0.1", TypeError), ("batch_size", 0, ValueError),
    ("probe_interval", 0, ValueError), ("ann_effort", -1, ValueError),
])
def test_invalid_config(field, value, error):
    values = dict(k=2, seed=0, max_steps=10, tolerance=0.0)
    values[field] = value
    with pytest.raises(error):
        FitConfig(**values)


@pytest.mark.parametrize("bad", [
    [[1, 2]], np.ones((2, 2), np.float64), np.ones((2, 2), np.int32),
    np.ones(2, np.float32), np.ones((0, 2), np.float32),
    np.ones((2, 0), np.float32), np.ones((2, 2, 1), np.float32),
    np.ones((4, 4), np.float32)[:, ::2],
    np.array([[np.nan, 0]], np.float32), np.array([[np.inf, 0]], np.float32),
])
def test_invalid_feature_arrays(bad):
    good = np.ones((2, 2), np.float32)
    with pytest.raises((TypeError, ValueError)):
        initialize_centroids(bad, 1, seed=0)
    with pytest.raises((TypeError, ValueError)):
        fit_cpu(bad, good, config())
    with pytest.raises((TypeError, ValueError)):
        fit_cpu(good, bad, config())
    with pytest.raises((TypeError, ValueError)):
        assign_exact(good, bad)


def test_inconsistent_inputs(two_groups):
    points, initial = two_groups
    for centroids, cfg in [(initial, config(k=1)),
                            (np.ones((5, 2), np.float32), config(k=5)),
                            (np.ones((2, 3), np.float32), config())]:
        with pytest.raises(ValueError):
            fit_cpu(points, centroids, cfg)
    with pytest.raises(TypeError):
        fit_cpu(points, initial, {})
    for kwargs in ({"batch_size": 2}, {"ann_effort": 4}):
        with pytest.raises(ValueError):
            fit_cpu(points, initial, config(**kwargs))
    with pytest.raises(ValueError):
        initialize_centroids(points, 5, seed=0)


@pytest.mark.parametrize("labels", [
    np.array([0, 0, 1, 1], np.int64), np.array([0, 0], np.int32),
    np.array([0, 0, 1, -1], np.int32), np.array([0, 0, 1, 2], np.int32),
    np.zeros((4, 1), np.int32), [0, 0, 1, 1],
])
def test_invalid_labels(two_groups, labels):
    with pytest.raises((TypeError, ValueError)):
        accumulate_cluster_stats(two_groups[0], labels, 2)


def test_contracts_import_without_numba_or_cuda():
    # A fresh interpreter proves shared definitions have no GPU/JIT dependency.
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-B", "-c",
                    "import sys; import pcap.contracts; "
                    "assert 'numba' not in sys.modules; assert 'numba.cuda' not in sys.modules"],
                   check=True, env=env)


@pytest.mark.parametrize("dimensions", [1, 3, 7])
@pytest.mark.parametrize("k", [1, 3, 8])
def test_independent_multistep_numpy_lloyd(dimensions, k):
    rng = np.random.default_rng(2026 + dimensions + k)
    points = rng.normal(size=(20, dimensions)).astype(np.float32)
    initial = points[:k].copy()
    expected = initial.copy()
    expected_objectives = [numpy_oracle(points, expected)[1].sum()]
    movements = [0.0]
    # Independent NumPy means and norms, without production assignment/update calls.
    for step in range(4):
        labels, _ = numpy_oracle(points, expected)
        updated = expected.copy()
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.astype(np.float64).mean(axis=0)
        shift = np.linalg.norm(updated.astype(np.float64) - expected, axis=1).max()
        expected = updated
        movements.append(shift)
        expected_objectives.append(numpy_oracle(points, expected)[1].sum())
        if shift <= 0.0:
            break
    result = fit_cpu(points, initial, config(k=k, max_steps=4))
    np.testing.assert_array_equal(result.centroids, expected)
    labels, distances = numpy_oracle(points, expected)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.sse, distances.sum(), rtol=1e-14)
    np.testing.assert_allclose([entry["sse"] for entry in result.history],
                               expected_objectives, rtol=1e-14)
    np.testing.assert_allclose([entry["centroid_displacement"] for entry in result.history],
                               movements, rtol=1e-14)


def test_final_sse_is_independent_of_history(two_groups):
    result = fit_cpu(*two_groups, config())
    assert result.metadata["final_exact_sse"] == 4.0
    result.history.append({"sse": 999.0, "scope": "probe"})
    assert result.sse == 4.0
    result.history.clear()
    assert result.sse == 4.0
    # Existing five-field construction works for other producers without history.
    shared = RunResult(result.centroids, result.labels, [], {}, {"final_exact_sse": 4.0})
    assert shared.sse == 4.0


@pytest.mark.parametrize("metadata", [
    {}, {"final_exact_sse": None}, {"final_exact_sse": np.nan},
    {"final_exact_sse": np.inf}, {"final_exact_sse": -1},
    {"final_exact_sse": True}, {"final_exact_sse": "4"},
])
def test_final_sse_rejects_missing_or_invalid_metric(two_groups, metadata):
    points, initial = two_groups
    result = RunResult(initial, np.zeros(len(points), np.int32), [{"sse": 4}], {}, metadata)
    with pytest.raises(ValueError, match="final_exact_sse"):
        _ = result.sse


def test_final_assignment_timing_is_a_subset(two_groups, monkeypatch):
    import pcap.kmeans_cpu as cpu

    # Deterministic clock boundaries verify accounting without timing thresholds.
    ticks = iter([0.0, 1.0, 3.0, 5.0, 6.0])
    monkeypatch.setattr(cpu, "perf_counter", lambda: next(ticks))
    original = cpu._final_exact_assignment
    calls = []

    def counted_final(points, centroids):
        calls.append(1)
        return original(points, centroids)

    monkeypatch.setattr(cpu, "_final_exact_assignment", counted_final)
    result = cpu.fit_cpu(*two_groups, config())
    assert calls == [1]
    assert result.timings == {"compute_s": 4.0, "wall_s": 6.0, "final_assignment_s": 2.0}
    assert result.sse == 4.0
