"""Real-device Mini-Batch CUDA tests. Run with scripts/run_cuda_tests.sh."""

import math
import numpy as np
import pytest
try:
    from numba import cuda
    from pcap.assignments_exact import assign_exact
    from pcap.minibatch import fit_minibatch_cpu
    from pcap.minibatch_cuda import fit_minibatch_cuda
except ImportError:
    pytest.skip("numba/CUDA required for CUDA Mini-Batch tests", allow_module_level=True)


@pytest.fixture
def real_cuda():
    assert not cuda.config.ENABLE_CUDASIM, "Real CUDA hardware is required"
    assert cuda.is_available(), "Run with scripts/run_cuda_tests.sh"
    assert cuda.get_current_device().compute_capability >= (6, 0)


def config(k, b, steps=10, seed=0, interval=5, tolerance=1e-5):
    return FitConfig(k=k, seed=seed, max_steps=steps, tolerance=tolerance,
                     batch_size=b, probe_interval=interval)


@pytest.mark.parametrize("n,d,k,b", [
    (64, 2, 2, 16),
    (128, 4, 3, 32),
    (256, 4, 4, 64),
    (100, 14, 5, 100),  # full batch (b == n)
])
def test_cuda_minibatch_matches_cpu(real_cuda, n, d, k, b):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(n, d)).astype(np.float32)
    initial = points[:k].copy()
    cfg = config(k, b, steps=15, seed=7, interval=5, tolerance=1e-4)

    saved_points, saved_initial = points.copy(), initial.copy()
    cpu = fit_minibatch_cpu(points, initial, cfg)
    gpu = fit_minibatch_cuda(points, initial, cfg)

    # Output array types
    assert gpu.centroids.dtype == np.float32
    assert gpu.labels.dtype == np.int32
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(initial, saved_initial)

    # Numerical consistency between CPU and GPU
    np.testing.assert_allclose(gpu.centroids, cpu.centroids, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        gpu.metadata["cluster_observation_counts"],
        cpu.metadata["cluster_observation_counts"],
    )
    assert gpu.metadata["steps"] == cpu.metadata["steps"]
    assert gpu.metadata["termination_reason"] == cpu.metadata["termination_reason"]

    # Final exact assignment verification
    exact = assign_exact(points, gpu.centroids)
    np.testing.assert_array_equal(gpu.labels, exact.labels)
    np.testing.assert_allclose(
        gpu.sse, float(np.sum(exact.squared_distances, dtype=np.float64)),
        rtol=1e-9, atol=1e-9,
    )
    np.testing.assert_allclose(gpu.sse, cpu.sse, rtol=1e-5, atol=1e-5)

    # Timings sanity
    timings = gpu.timings
    assert all(np.isfinite(v) and v >= 0 for v in timings.values())
    assert timings["final_assignment_s"] <= timings["compute_s"] <= timings["wall_s"]
    assert timings["scalar_transfer_s"] <= timings["compute_s"]
    assert timings["transfer_s"] == pytest.approx(
        timings["upload_s"] + timings["scalar_transfer_s"] + timings["download_s"]
    )


def test_cuda_minibatch_convergence_tolerance(real_cuda, two_groups):
    points, initial = two_groups
    # With two well-separated clusters, it should converge quickly
    cfg = FitConfig(k=2, seed=0, max_steps=50, tolerance=1.0,
                    batch_size=4, probe_interval=2)
    result = fit_minibatch_cuda(points, initial, cfg)
    assert result.metadata["termination_reason"] == "tolerance"
    assert result.metadata["convergence_satisfied"] is True
    assert result.metadata["steps"] < 50


def test_cuda_minibatch_empty_cluster_retained(real_cuda):
    # Cluster 2 is far away and receives 0 observations
    points = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]], dtype=np.float32)
    initial = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    cfg = FitConfig(k=2, seed=0, max_steps=3, tolerance=0.0,
                    batch_size=2, probe_interval=10)
    result = fit_minibatch_cuda(points, initial, cfg)
    # Cluster 1 must retain its initial centroid
    np.testing.assert_array_equal(result.centroids[1], initial[1])
    assert result.metadata["cluster_observation_counts"][1] == 0


def test_cuda_minibatch_reproducibility(real_cuda, four_features):
    points = four_features
    initial = points[:3].copy()
    cfg = FitConfig(k=3, seed=99, max_steps=10, tolerance=1e-5,
                    batch_size=16, probe_interval=3)
    run1 = fit_minibatch_cuda(points, initial, cfg)
    run2 = fit_minibatch_cuda(points, initial, cfg)
    np.testing.assert_array_equal(run1.centroids, run2.centroids)
    np.testing.assert_array_equal(run1.labels, run2.labels)
    assert run1.sse == run2.sse


@pytest.mark.parametrize("kwargs", [
    {"batch_size": None},
    {"batch_size": 100},  # > n (n=20)
    {"ann_effort": 16},   # must be None for fixed exact
    {"tolerance": -1.0},
])
def test_cuda_minibatch_invalid_config(real_cuda, kwargs):
    points = np.zeros((20, 2), dtype=np.float32)
    initial = np.zeros((2, 2), dtype=np.float32)
    base = {"k": 2, "seed": 0, "max_steps": 5, "tolerance": 1e-4, "batch_size": 10}
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        cfg = FitConfig(**base)
        fit_minibatch_cuda(points, initial, cfg)
