"""Real-device correctness and lifecycle tests for the isolated ANN prototype."""
import numpy as np
import pytest

from numba import cuda

from pcap.assignments_ann_cuda import ProjectionCUDAIndex
from pcap.assignments_ann_multi_cuda import MultiProjectionCUDAIndex
from pcap.assignments_exact import assign_exact


pytestmark = pytest.mark.skipif(
    cuda.config.ENABLE_CUDASIM or not cuda.is_available(),
    reason="requires a real CUDA device",
)


@pytest.mark.parametrize("k", [16, 64, 256])
@pytest.mark.parametrize("d", [4, 14])
def test_multi_projection_returns_selected_distance_and_valid_effort(k, d):
    rng = np.random.default_rng(k * 17 + d)
    X = np.ascontiguousarray(rng.normal(size=(513, d)).astype(np.float32))
    C = np.ascontiguousarray(X[rng.choice(len(X), size=k, replace=False)].copy())
    exact = assign_exact(X, C)
    backend = MultiProjectionCUDAIndex(seed=7, ann_effort=8)
    backend.prepare(C)
    actual = backend.assign(X, C)
    # Distances are exact for the selected IDs; approximate labels are not
    # required to match exhaustive nearest-centroid assignment universally.
    selected = np.sum((X.astype(np.float64) - C[actual.labels].astype(np.float64)) ** 2,
                      axis=1, dtype=np.float64)
    np.testing.assert_allclose(actual.squared_distances, selected, rtol=1e-12, atol=1e-12)
    assert actual.labels.dtype == np.int32
    assert actual.labels.flags.c_contiguous
    assert backend.last_candidate_distance_evaluations <= len(X) * 8
    assert backend.last_candidate_distance_evaluations >= len(X)
    if k > 8:
        assert backend.search_mode(8) == "approximate_multi_projection"
    assert exact.squared_distances.shape == actual.squared_distances.shape


def test_multi_projection_reports_explicit_exact_fallback_and_tie_ids():
    X = np.ascontiguousarray(np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32))
    C = np.ascontiguousarray(np.repeat(X[:1], 16, axis=0))
    backend = MultiProjectionCUDAIndex(seed=0, ann_effort=16)
    backend.prepare(C)
    result = backend.assign(X, C)
    assert backend.search_mode(16) == "exact_budget_fallback"
    np.testing.assert_array_equal(result.labels, np.zeros(2, dtype=np.int32))
    assert backend.last_candidate_distance_evaluations == len(X) * len(C)


def test_multi_projection_refresh_rejects_stale_versions():
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(rng.normal(size=(128, 4)).astype(np.float32))
    C0 = np.ascontiguousarray(X[:16].copy())
    C1 = np.ascontiguousarray(C0 + np.float32(0.1))
    backend = MultiProjectionCUDAIndex(seed=3, ann_effort=8)
    backend.prepare(C0)
    points = cuda.to_device(X)
    indices = cuda.to_device(np.arange(len(X), dtype=np.int32))
    labels = cuda.device_array(len(X), dtype=np.int32)
    distances = cuda.device_array(len(X), dtype=np.float64)
    counts = cuda.device_array(len(X), dtype=np.int32)
    with pytest.raises(ValueError, match="stale"):
        backend.assign_device(points, indices, labels, distances, counts, version=1)
    device_C1 = cuda.to_device(C1)
    backend.refresh_device(device_C1, version=1)
    assert backend.version == 1
    with pytest.raises(ValueError, match="stale"):
        backend.assign_device(points, indices, labels, distances, counts, version=0)
    backend.assign_device(points, indices, labels, distances, counts, version=1)
    refreshed = backend.assign(X, C1)
    exact_refreshed = assign_exact(X, C1)
    assert refreshed.labels.dtype == np.int32
    selected = np.sum((X.astype(np.float64) - C1[refreshed.labels].astype(np.float64)) ** 2,
                      axis=1, dtype=np.float64)
    np.testing.assert_allclose(refreshed.squared_distances, selected,
                               rtol=1e-12, atol=1e-12)
    assert np.all(refreshed.squared_distances >= exact_refreshed.squared_distances - 1e-12)
    assert backend.timings["rebuild_count"] == 1


def test_multi_projection_effort_validation_and_deterministic_replay():
    rng = np.random.default_rng(91)
    X = np.ascontiguousarray(rng.normal(size=(257, 4)).astype(np.float32))
    C = np.ascontiguousarray(X[:64].copy())
    first = MultiProjectionCUDAIndex(seed=11, ann_effort=16)
    second = MultiProjectionCUDAIndex(seed=11, ann_effort=16)
    first.prepare(C)
    second.prepare(C)
    a = first.assign(X, C)
    b = second.assign(X, C)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.squared_distances, b.squared_distances)
    with pytest.raises(ValueError, match="multiple of"):
        first.assign(X, C, effort=6)
    with pytest.raises((TypeError, ValueError)):
        first.assign(X, C, effort=8.5)
    with pytest.raises(ValueError, match="<= 32"):
        MultiProjectionCUDAIndex(seed=1, ann_effort=36)


def test_existing_projection_reference_remains_unchanged_and_operational():
    rng = np.random.default_rng(19)
    X = np.ascontiguousarray(rng.normal(size=(128, 4)).astype(np.float32))
    C = np.ascontiguousarray(X[:32].copy())
    backend = ProjectionCUDAIndex(seed=2, ann_effort=8)
    backend.prepare(C)
    result = backend.assign(X, C)
    assert result.labels.shape == (len(X),)
    assert np.isfinite(result.squared_distances).all()
