"""Real-device M2 tests. Run with scripts/run_cuda_tests.sh tests."""

import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numba import cuda

from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.kmeans_cpu import fit_cpu
from pcap.kmeans_cuda import fit_cuda
import pcap.kmeans_cuda as cuda_impl


@pytest.fixture
def real_cuda():
    assert not cuda.config.ENABLE_CUDASIM, "Real CUDA hardware is required"
    assert cuda.is_available(), "Run with scripts/run_cuda_tests.sh"
    assert cuda.get_current_device().compute_capability >= (6, 0)


def compare(points, initial, config):
    saved_points, saved_initial = points.copy(), initial.copy()
    cpu = fit_cpu(points, initial, config)
    gpu = fit_cuda(points, initial, config)
    assert gpu.centroids.dtype == np.float32
    assert gpu.labels.dtype == np.int32
    np.testing.assert_allclose(gpu.centroids, cpu.centroids, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(gpu.labels, cpu.labels)
    np.testing.assert_allclose(gpu.sse, cpu.sse, rtol=1e-10, atol=1e-10)
    exact = assign_exact(points, gpu.centroids)
    np.testing.assert_array_equal(gpu.labels, exact.labels)
    np.testing.assert_allclose(gpu.sse, exact.squared_distances.sum(), rtol=1e-12, atol=1e-12)
    objectives = np.array([entry["sse"] for entry in gpu.history])
    assert np.all(np.diff(objectives) <= 1e-10 * np.maximum(1, objectives[:-1]))
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(initial, saved_initial)
    assert not np.shares_memory(gpu.centroids, initial)
    assert gpu.metadata["termination_reason"] == cpu.metadata["termination_reason"]
    assert gpu.metadata["steps"] == cpu.metadata["steps"]
    assert gpu.metadata["final_exact_sse"] == gpu.history[-1]["sse"]
    timings = gpu.timings
    assert all(np.isfinite(value) and value >= 0 for value in timings.values())
    assert timings["final_assignment_s"] <= timings["compute_s"] <= timings["wall_s"]
    assert timings["kernel_s"] <= timings["compute_s"] + 0.01
    assert timings["upload_s"] <= timings["setup_s"]
    assert timings["transfer_s"] == pytest.approx(
        timings["upload_s"] + timings["scalar_transfer_s"] + timings["download_s"])
    delta = float(np.max(np.abs(gpu.centroids.astype(np.float64) - cpu.centroids)))
    print(f"N={len(points)} D={points.shape[1]} K={config.k}: "
          f"max centroid delta={delta:.3g}, SSE delta={gpu.sse - cpu.sse:.3g}")
    return gpu


@pytest.mark.parametrize("n,d,k,seed", [
    (1, 1, 1, 0), (17, 3, 1, 1), (257, 1, 3, 2),
    (513, 4, 4, 3), (4099, 7, 8, 4), (73, 16, 16, 5),
    (1025, 3, 257, 6),
])
def test_gpu_matches_cpu_across_shapes(real_cuda, n, d, k, seed):
    points = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    initial = initialize_centroids(points, k, seed=seed)
    compare(points, initial, FitConfig(k, seed, 8, 1e-6))


@pytest.mark.parametrize("side", [512, 1024])
def test_image_sized_synthetic_inputs(real_cuda, side):
    n, d, k = side * side, 4, 4
    rng = np.random.default_rng(42)
    centers = np.arange(k, dtype=np.float32)[:, None] * np.ones((1, d), np.float32) * 5
    points = (centers[np.arange(n) % k] + rng.normal(0, 0.1, (n, d))).astype(np.float32)
    compare(points, centers, FitConfig(k, 42, 3, 1e-6))


def test_gpu_known_solution(real_cuda, two_groups):
    result = compare(*two_groups, FitConfig(2, 42, 30, 0.0))
    np.testing.assert_array_equal(result.centroids, [[0, 1], [10, 11]])
    assert result.sse == 4.0
    result.history.append({"sse": 9999, "scope": "probe"})
    assert result.sse == 4.0


@pytest.mark.parametrize("case", ["ties", "empty", "identical", "reassignment", "tolerance"])
def test_gpu_edge_cases(real_cuda, case):
    steps, tolerance = 10, 0.0
    if case == "ties":
        points = np.array([[1], [3], [-1]], np.float32)
        initial = np.array([[0], [2], [2]], np.float32)
    elif case == "empty":
        points = np.array([[0], [2], [10], [12]], np.float32)
        initial = np.array([[0], [10], [100]], np.float32)
    elif case == "identical":
        points = np.ones((33, 7), np.float32)
        initial = np.ones((5, 7), np.float32)
    elif case == "reassignment":
        points = np.array([[0], [2], [3], [10]], np.float32)
        initial = np.array([[0], [3]], np.float32)
        steps = 1
    else:
        points = np.array([[0, 0], [2, 2]], np.float32)
        initial = points[:1].copy()
        tolerance = 2.0
    result = compare(points, initial, FitConfig(len(initial), 0, steps, tolerance))
    if case == "ties":
        assert result.labels[0] == 0
    elif case == "empty":
        np.testing.assert_array_equal(result.centroids[-1], [100])
    elif case == "identical":
        np.testing.assert_array_equal(result.labels, np.zeros(len(points), np.int32))
    elif case == "reassignment":
        np.testing.assert_array_equal(result.labels, [0, 0, 1, 1])


def test_gpu_repeated_runs(real_cuda, four_features):
    initial = initialize_centroids(four_features, 3, seed=17)
    config = FitConfig(3, 17, 20, 1e-6)
    first = compare(four_features, initial, config)
    second = fit_cuda(four_features, initial, config)
    np.testing.assert_allclose(first.centroids, second.centroids, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.sse == pytest.approx(second.sse, rel=1e-12)


@pytest.mark.parametrize("bad", [
    [[1, 2]], np.ones((2, 2), np.float64), np.ones(2, np.float32),
    np.ones((0, 2), np.float32), np.ones((2, 0), np.float32),
    np.ones((4, 4), np.float32)[:, ::2],
    np.array([[np.nan, 0]], np.float32), np.array([[np.inf, 0]], np.float32),
])
def test_gpu_input_validation_before_device_access(bad):
    good = np.ones((2, 2), np.float32)
    config = FitConfig(2, 0, 10, 0.0)
    for points, initial in ((bad, good), (good, bad)):
        with pytest.raises((TypeError, ValueError)):
            fit_cuda(points, initial, config)


def test_gpu_config_consistency():
    points = np.ones((4, 2), np.float32)
    initial = points[:2].copy()
    with pytest.raises(TypeError):
        fit_cuda(points, initial, {})
    for centroids, config in [
        (initial, FitConfig(1, 0, 10, 0.0)),
        (np.ones((5, 2), np.float32), FitConfig(5, 0, 10, 0.0)),
        (np.ones((2, 3), np.float32), FitConfig(2, 0, 10, 0.0)),
        (initial, FitConfig(2, 0, 10, 0.0, batch_size=2)),
        (initial, FitConfig(2, 0, 10, 0.0, ann_effort=4)),
    ]:
        with pytest.raises(ValueError):
            fit_cuda(points, centroids, config)


def test_cpu_imports_with_cuda_blocked():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    source = """
import importlib.abc
import sys
class BlockCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numba.cuda' or fullname.startswith('numba.cuda.'):
            raise ImportError('CUDA intentionally blocked')
sys.meta_path.insert(0, BlockCUDA())
import pcap.contracts
import pcap.initialization
import pcap.assignments_exact
import pcap.kmeans_cpu
assert 'numba.cuda' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", source], env=env, check=True)


@pytest.mark.parametrize("large", [False, True], ids=["K257-last-centroid", "grid-stride-both-stages"])
def test_direct_movement_reduction(real_cuda, large):
    threads = cuda_impl._THREADS
    grid_span = threads * cuda_impl._MAX_BLOCKS
    k = 2 * grid_span - 17 if large else 257
    max_index = k - 1
    displacements = np.linspace(0.0, 1.0, k, dtype=np.float64)
    displacements[max_index] = 8.0
    squared_shifts = displacements ** 2
    blocks = cuda_impl._blocks(k)
    if large:
        assert k > blocks * threads  # First-stage threads must visit a second tile.
        assert max_index >= blocks * threads
        # The winning partial also lies beyond the second stage's first 256 values.
        assert (max_index % (blocks * threads)) // threads >= threads
    else:
        assert max_index == 256
        assert blocks == 2
    device_values = cuda.to_device(squared_shifts)
    partials = cuda.device_array(blocks, np.float64)
    scalar = cuda.device_array(1, np.float64)
    cuda_impl._reduce(device_values, partials, scalar, True)
    cuda.synchronize()
    actual_squared = scalar.copy_to_host()[0]
    # Max selects an existing nonnegative float64 value: no additive error is
    # introduced, so exact selection is appropriate (not a reproducible-sum claim).
    np.testing.assert_equal(actual_squared, np.max(squared_shifts))
    np.testing.assert_allclose(math.sqrt(actual_squared), np.max(displacements), rtol=1e-15)


def _stress_inputs(case):
    rng = np.random.default_rng(812)
    if case == "contention-one":
        points = rng.normal(0.3, 0.2, (65_539, 4)).astype(np.float32)
        initial = np.zeros((1, 4), np.float32)
    elif case == "contention-two":
        # Most atomic operations hit one centroid, with a small second group.
        points = rng.normal(-2.0, 0.1, (65_539, 3)).astype(np.float32)
        points[-257:] += 4.0
        rng.shuffle(points)
        initial = np.array([[-2.0] * 3, [2.0] * 3], np.float32)
    elif case == "cancellation":
        # Small residuals mixed with opposite-signed values nine orders larger.
        pattern = np.array([[1e8, -1e8, 1e-3],
                            [0.1, 0.3, -1e-3],
                            [-1e8, 1e8, 0.25]], np.float32)
        points = np.tile(pattern, (16_385, 1))
        rng.shuffle(points)
        initial = np.zeros((1, 3), np.float32)
    else:
        # Squared distances exceed float32 range but remain finite in float64.
        points = rng.normal(-2.0, 0.1, (8193, 3)).astype(np.float32)
        points[4096:] += 4.0
        points *= np.float32(1e30)
        initial = np.array([[-2e30] * 3, [2e30] * 3], np.float32)
    return np.ascontiguousarray(points), initial


def _numpy_assignment(points, centroids):
    distances = np.sum((points.astype(np.float64)[:, None, :]
                        - centroids.astype(np.float64)[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, distances[np.arange(len(points)), labels]


def _reference_stats(points, labels, k):
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    # fsum is independent of the production sequential/atomic summation order.
    sums = np.array([[math.fsum(points[labels == j, d].astype(np.float64))
                      for d in range(points.shape[1])] for j in range(k)], dtype=np.float64)
    abs_sums = np.array([[math.fsum(np.abs(points[labels == j, d].astype(np.float64)))
                          for d in range(points.shape[1])] for j in range(k)], dtype=np.float64)
    # Standard forward bound for summing n finite terms in arbitrary order:
    # gamma_n * sum(abs(x)), allowing absolute rather than relative error when
    # positive/negative values cancel. float32 inputs convert exactly to float64.
    eps64 = np.finfo(np.float64).eps
    n_eps = len(points) * eps64
    gamma_n = n_eps / (1.0 - n_eps)
    sum_bounds = gamma_n * abs_sums
    return sums, counts, sum_bounds


@pytest.mark.parametrize("case", ["contention-one", "contention-two", "cancellation", "large-finite"])
def test_atomic_and_lloyd_numerical_stress(real_cuda, case):
    points, initial = _stress_inputs(case)
    k = len(initial)
    initial_labels, _ = _numpy_assignment(points, initial)
    expected_sums, expected_counts, sum_bounds = _reference_stats(points, initial_labels, k)
    d_points = cuda.to_device(points)
    d_labels = cuda.to_device(initial_labels.astype(np.int32))
    d_sums = cuda.device_array((k, points.shape[1]), np.float64)
    d_counts = cuda.device_array(k, np.int64)
    actual_sums = []
    for _ in range(3):
        cuda_impl._reset_stats_kernel[cuda_impl._blocks(k * points.shape[1]), cuda_impl._THREADS](
            d_sums, d_counts)
        cuda_impl._accumulate_kernel[cuda_impl._blocks(len(points)), cuda_impl._THREADS](
            d_points, d_labels, d_sums, d_counts)
        cuda.synchronize()
        sums = d_sums.copy_to_host()
        np.testing.assert_array_equal(d_counts.copy_to_host(), expected_counts)
        assert np.all(np.abs(sums - expected_sums) <= sum_bounds), (case, sums, expected_sums)
        actual_sums.append(sums)
    # Repeated sums may differ: each must meet the independent error bound.
    assert np.all(np.abs(actual_sums[0] - actual_sums[-1]) <= 2 * sum_bounds)

    cfg = FitConfig(k, 812, 6, 1e-6)
    cpu = fit_cpu(points, initial, cfg)
    runs = [fit_cuda(points, initial, cfg) for _ in range(3)]
    max_relative_sse_error = 0.0
    max_centroid_delta = 0.0
    for result in [cpu, *runs]:
        labels, distances = _numpy_assignment(points, result.centroids)
        np.testing.assert_array_equal(result.labels, labels)
        assert np.isfinite(result.centroids).all() and np.isfinite(result.sse)
        reference_sse = math.fsum(distances)
        # n <= 65539 gives gamma_n < 1.5e-11. 1e-10 allows the positive SSE
        # reduction's rounding plus distance arithmetic, with ample margin.
        np.testing.assert_allclose(result.sse, reference_sse, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(result.labels, cpu.labels)
        sums, counts, bounds = _reference_stats(points, result.labels, k)
        means = sums / counts[:, None]
        # Sum error divided by count, plus float32 rounding of stored centroids.
        mean_bounds = bounds / counts[:, None] + np.finfo(np.float32).eps * np.abs(means)
        mean_bounds += np.finfo(np.float32).tiny
        assert np.all(np.abs(result.centroids.astype(np.float64) - means) <= mean_bounds)
        max_relative_sse_error = max(max_relative_sse_error,
                                     abs(result.sse - cpu.sse) / max(cpu.sse, 1.0))
        max_centroid_delta = max(max_centroid_delta,
                                 float(np.max(np.abs(result.centroids.astype(np.float64)
                                                     - cpu.centroids))))
    # Compare repeated outputs with tolerances, not bitwise floating equality.
    np.testing.assert_allclose(runs[0].centroids, runs[-1].centroids,
                               rtol=2e-6, atol=float(np.max(2 * mean_bounds)))
    np.testing.assert_allclose(runs[0].sse, runs[-1].sse, rtol=1e-10, atol=1e-12)
    if case == "large-finite":
        assert reference_sse > float(np.finfo(np.float32).max)
        assert np.max(distances) > float(np.finfo(np.float32).max)
    print(f"Stress {case}, 3 GPU runs: max centroid delta vs CPU={max_centroid_delta:.3g}, "
          f"max relative SSE delta vs CPU={max_relative_sse_error:.3g}")


def test_direct_near_and_exact_tie_assignments(real_cuda):
    one = np.float32(1.0)
    below = np.nextafter(one, np.float32(0.0))
    above = np.nextafter(one, np.float32(2.0))
    points = np.array([[below, 0], [one, 0], [above, 0], [2, 0], [1, 3]], np.float32)
    # Duplicate centroid 2 also checks lowest-index selection away from index 0.
    initial = np.array([[0, 0], [2, 0], [2, 0]], np.float32)
    expected_labels, expected_distances = _numpy_assignment(points, initial)
    np.testing.assert_array_equal(expected_labels, [0, 0, 1, 1, 0])
    cpu = assign_exact(points, initial)
    np.testing.assert_array_equal(cpu.labels, expected_labels)
    d_points, d_centroids = cuda.to_device(points), cuda.to_device(initial)
    labels = cuda.device_array(len(points), np.int32)
    distances = cuda.device_array(len(points), np.float64)
    for _ in range(3):
        cuda_impl._assign_kernel[cuda_impl._blocks(len(points)), cuda_impl._THREADS](
            d_points, d_centroids, labels, distances)
        cuda.synchronize()
        np.testing.assert_array_equal(labels.copy_to_host(), expected_labels)
        # Small D=2 bounds distance-rounding error to a few double-precision ulps.
        np.testing.assert_allclose(distances.copy_to_host(), expected_distances, rtol=1e-14, atol=1e-15)
    # Full Lloyd path also remains consistent after these boundary assignments.
    compare(points, initial, FitConfig(3, 0, 8, 1e-6))
