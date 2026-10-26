"""Real-device tests for the isolated decayed-statistics experiment."""

import numpy as np
import pytest
from numba import cuda

from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig
from pcap.adaptive_v2_controller import AdaptiveV2Policy
from pcap.minibatch_decayed_cuda import (
    fit_adaptive_decayed_minibatch_cuda,
    fit_decayed_minibatch_cuda,
)


@pytest.fixture
def real_cuda():
    assert not cuda.config.ENABLE_CUDASIM, "Real CUDA hardware is required"
    assert cuda.is_available(), "Run with scripts/run_cuda_tests.sh"


def config(k, batch, steps=8, seed=3, probe=2, tolerance=0.0):
    return FitConfig(k, seed, steps, tolerance, batch_size=batch, probe_interval=probe)


def numpy_decayed_reference(points, initial, samples):
    """Independent float64 nearest assignment and decayed-statistics reference."""
    n, _ = points.shape
    k, d = initial.shape
    centroids = initial.copy()
    sums = np.zeros((k, d), np.float64)
    weights = np.zeros(k, np.float64)
    movements = []
    for indices in samples:
        x = points[indices].astype(np.float64)
        delta = x[:, None, :] - centroids.astype(np.float64)[None, :, :]
        squared = np.sum(delta * delta, axis=2, dtype=np.float64)
        labels = np.argmin(squared, axis=1)
        batch_sums = np.zeros((k, d), np.float64)
        batch_counts = np.zeros(k, np.int64)
        for point, label in zip(x, labels):
            batch_sums[label] += point
            batch_counts[label] += 1
        decay = 0.5 ** (len(indices) / (10.0 * n))
        sums = decay * sums + batch_sums
        weights = decay * weights + batch_counts
        old = centroids.copy()
        active = weights > 0
        centroids[active] = (sums[active] / weights[active, None]).astype(np.float32)
        movements.append(float(np.sqrt(np.max(np.sum(
            (centroids.astype(np.float64) - old.astype(np.float64))**2, axis=1)))))
    return centroids, sums, weights, movements


def samples_for(n, batch, steps, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(n, size=batch, replace=False) for _ in range(steps)],
                      dtype=np.int32)


def test_fixed_decayed_cuda_matches_independent_numpy_reference(real_cuda):
    rng = np.random.default_rng(721)
    points = np.ascontiguousarray(np.vstack((
        rng.normal((-3, 1, 0, 4), .2, (40, 4)),
        rng.normal((2, -2, 3, 1), .3, (40, 4)),
        rng.normal((6, 5, -1, -3), .1, (40, 4)),
    )), dtype=np.float32)
    initial = points[[0, 40, 80]].copy()
    cfg = config(3, 17, steps=9, seed=19, probe=3)
    samples = samples_for(len(points), cfg.batch_size, cfg.max_steps, cfg.seed)
    expected_c, expected_sums, expected_weights, _ = numpy_decayed_reference(points, initial, samples)
    actual = fit_decayed_minibatch_cuda(points, initial, cfg, quality_interval=3,
                                        batch_indices=samples)
    assert actual.metadata["execution_device"] == "cuda"
    np.testing.assert_allclose(actual.centroids, expected_c, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual.metadata["effective_cluster_counts"], expected_weights,
                               rtol=1e-12, atol=1e-12)
    assert actual.centroids.dtype == np.float32
    assert actual.labels.dtype == np.int32 and actual.labels.flags.c_contiguous
    assert actual.metadata["accumulation_dtype"] == "float64"
    assert actual.metadata["quality_history"][-1]["step"] == cfg.max_steps
    exact = assign_exact(points, actual.centroids)
    np.testing.assert_array_equal(actual.labels, exact.labels)
    assert actual.sse == pytest.approx(float(exact.squared_distances.sum(dtype=np.float64)),
                                       rel=1e-12, abs=1e-12)
    assert np.isfinite(expected_sums).all()
    assert all(np.isfinite(v) and v >= 0 for v in actual.timings.values())


def test_decay_uses_actual_batch_size_and_keeps_statistics_on_device(real_cuda):
    points = np.ascontiguousarray(np.arange(96, dtype=np.float32).reshape(48, 2))
    initial = points[[0, -1]].copy()
    cfg = config(2, 12, steps=4, seed=4, probe=2)
    result = fit_decayed_minibatch_cuda(points, initial, cfg, quality_interval=2)
    expected = 0.5 ** (cfg.batch_size / (10.0 * len(points)))
    assert result.metadata["decay_rule"].startswith("lambda_t=0.5**")
    assert all(row["forgetting_factor"] == pytest.approx(expected) for row in result.history)
    assert result.metadata["sampling_policy"].startswith("uniform without replacement")
    assert result.metadata["estimated_device_buffer_bytes"] > points.nbytes


def test_empty_clusters_retain_initial_centroids(real_cuda):
    points = np.array([[0, 0], [.1, 0], [0, .1], [-.1, -.1]], np.float32)
    initial = np.array([[0, 0], [100, 100], [-100, 50]], np.float32)
    result = fit_decayed_minibatch_cuda(points, initial, config(3, 4, steps=3, probe=10))
    np.testing.assert_array_equal(result.centroids[1:], initial[1:])
    assert result.metadata["effective_cluster_counts"][1:] == [0.0, 0.0]


def test_repeated_fixed_runs_reproduce_sampling_and_quality(real_cuda):
    rng = np.random.default_rng(32)
    points = np.ascontiguousarray(rng.normal(size=(96, 14)), dtype=np.float32)
    initial = points[[2, 30, 62, 90]].copy()
    cfg = config(4, 15, steps=12, seed=9, probe=3)
    left = fit_decayed_minibatch_cuda(points, initial, cfg, quality_interval=4)
    right = fit_decayed_minibatch_cuda(points, initial, cfg, quality_interval=4)
    np.testing.assert_array_equal(left.labels, right.labels)
    np.testing.assert_allclose(left.centroids, right.centroids, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(left.sse, right.sse, rtol=1e-12, atol=1e-12)
    assert [r["batch_sse_before_update"] for r in left.history] == pytest.approx(
        [r["batch_sse_before_update"] for r in right.history], rel=1e-12, abs=1e-12)


def test_large_finite_mixed_values_remain_finite_and_match_exact_final_sse(real_cuda):
    points = np.array([[-1e15, 2e15], [1e15, -2e15], [-1e15, -2e15],
                       [1e15, 2e15], [0, 0], [3e14, -4e14]], np.float32)
    initial = points[[0, 1]].copy()
    result = fit_decayed_minibatch_cuda(points, initial, config(2, 4, steps=5, probe=2))
    assert np.isfinite(result.centroids).all()
    assert np.isfinite(result.sse)
    exact = assign_exact(points, result.centroids)
    assert result.sse == pytest.approx(float(exact.squared_distances.sum(dtype=np.float64)),
                                       rel=2e-12)


def test_exact_ties_final_labels_follow_lowest_centroid_index(real_cuda):
    points = np.array([[-1, 0], [1, 0], [0, 0], [0, 0]], np.float32)
    initial = np.array([[-1, 0], [1, 0]], np.float32)
    result = fit_decayed_minibatch_cuda(points, initial, config(2, 4, steps=1, probe=10))
    expected = assign_exact(points, result.centroids)
    np.testing.assert_array_equal(result.labels, expected.labels)
    assert result.labels.dtype == np.int32


def test_adaptive_v2_changes_batch_without_resetting_effective_counts(real_cuda):
    points = np.zeros((32, 2), np.float32)
    initial = np.zeros((1, 2), np.float32)
    cfg = config(1, 2, steps=6, seed=5, probe=1)
    policy = AdaptiveV2Policy(minimum=2, maximum=8, initial_batch=2,
                              required_checkpoints=1,
                              stagnation_improvement=0.001,
                              progress_improvement=0.01)
    result = fit_adaptive_decayed_minibatch_cuda(
        points, initial, cfg, policy=policy, quality_interval=2)
    sizes = [row["batch_size"] for row in result.history]
    assert sizes[:3] == [2, 4, 8]
    effective = 0.0
    for row in result.history:
        effective = row["forgetting_factor"] * effective + row["batch_size"]
    assert result.metadata["effective_cluster_counts"][0] == pytest.approx(effective, rel=1e-12)
    assert result.metadata["controller_calls"] > 0
    assert result.metadata["controller_signals"].find("full-data") == -1
    assert "target_sse" not in result.metadata
    assert result.metadata["execution_device"] == "cuda"


def test_adaptive_v2_does_not_accept_evaluation_target(real_cuda):
    import inspect
    from pcap.minibatch_decayed_cuda import fit_adaptive_decayed_minibatch_cuda

    assert "target_sse" not in inspect.signature(fit_adaptive_decayed_minibatch_cuda).parameters
    assert "reference_sse" not in inspect.signature(fit_adaptive_decayed_minibatch_cuda).parameters


@pytest.mark.parametrize("bad_samples", [
    np.zeros((3, 2), dtype=np.float32),
    np.zeros((2, 3), dtype=np.int32),
    np.array([[0, 1], [2, 99]], dtype=np.int32),
])
def test_replay_index_validation(real_cuda, bad_samples):
    points = np.zeros((8, 2), np.float32)
    initial = np.zeros((2, 2), np.float32)
    cfg = config(2, 2, steps=2)
    with pytest.raises(ValueError):
        fit_decayed_minibatch_cuda(points, initial, cfg, batch_indices=bad_samples)
