"""HNSW feasibility correctness tests; fake-index tests are not ANN evidence."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pcap.assignments_ann import HNSWCPUBackend
from pcap.assignments_exact import assign_exact


@pytest.fixture
def fake_hnsw(monkeypatch):
    """Exact stub solely for adapter lifecycle, batching and accounting tests."""
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

    monkeypatch.setitem(sys.modules, 'hnswlib', SimpleNamespace(Index=Index))
    return instances


@pytest.mark.parametrize('kwargs', [{'seed': -1}, {'seed': True}, {'m': 1},
                                  {'ef_construction': 2}, {'ann_effort': 0},
                                  {'query_batch_size': 0}, {'seed': 2**32}])
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        HNSWCPUBackend(**kwargs)


@pytest.mark.parametrize('bad', [np.zeros((2, 4), np.float64),
                                np.zeros((0, 4), np.float32),
                                np.full((2, 4), np.nan, np.float32),
                                np.zeros((4, 4), np.float32)[:, ::2],
                                np.full((2, 4), 1e30, np.float32)])
def test_invalid_centroids_without_dependency(bad):
    with pytest.raises((TypeError, ValueError)):
        HNSWCPUBackend().prepare(bad)


def test_optional_dependency_is_loaded_only_on_prepare(monkeypatch):
    import pcap.assignments_ann as module
    original = module.importlib.import_module

    def blocked(name, package=None):
        if name == 'hnswlib':
            raise ImportError('intentionally unavailable')
        return original(name, package)

    monkeypatch.setattr(module.importlib, 'import_module', blocked)
    backend = HNSWCPUBackend()
    with pytest.raises(ImportError, match='optional hnswlib'):
        backend.prepare(np.zeros((1, 4), np.float32))
    # Exact CPU remains usable without HNSW; contracts have no optional imports.
    assert assign_exact(np.zeros((2, 4), np.float32), np.zeros((1, 4), np.float32)).labels.tolist() == [0, 0]


def test_lifecycle_batching_effort_and_timing(fake_hnsw, monkeypatch):
    import pcap.assignments_ann as module
    ticks = iter(range(20))
    monkeypatch.setattr(module, 'perf_counter', lambda: next(ticks))
    centers = np.array([[0, 0], [10, 10]], np.float32)
    points = np.zeros((11, 2), np.float32)
    backend = HNSWCPUBackend(seed=7, ann_effort=4, query_batch_size=4)
    with pytest.raises(RuntimeError):
        backend.assign(points, centers)
    with pytest.raises(RuntimeError):
        backend.refresh(centers)
    backend.prepare(centers)
    index = fake_hnsw[-1]
    assert index.settings == dict(max_elements=2, M=8, ef_construction=100, random_seed=7)
    first = backend.assign(points, centers, effort=8)
    assert index.ef == 8 and index.batches == [4, 4, 3]
    assert first.query_time_s == first.index_update_time_s == 1.0
    assert backend.assign(points, centers).index_update_time_s == 0
    changed = centers.copy(); changed[0] = 3
    with pytest.raises(ValueError, match='stale'):
        backend.assign(points, changed)
    centers[0] = 3  # Mutation of caller array cannot mutate the owned snapshot.
    with pytest.raises(ValueError, match='stale'):
        backend.assign(points, centers)
    backend.refresh(changed)
    assert backend.timings['pending_index_update_s'] == 1
    refreshed = backend.assign(points, changed)
    assert refreshed.index_update_time_s == 1
    assert backend.timings == dict(initial_build_s=1.0, rebuild_s=1.0, query_s=3.0,
                                  initial_build_count=1, rebuild_count=1, query_count=3,
                                  pending_index_update_s=0.0)
    ledger = backend.timings; ledger['query_s'] = -1
    assert backend.timings['query_s'] == 3
    assert refreshed.labels.dtype == np.int32 and refreshed.labels.flags.c_contiguous
    assert refreshed.squared_distances.dtype == np.float64 and refreshed.squared_distances.flags.c_contiguous


def test_failed_query_keeps_pending_maintenance(fake_hnsw, monkeypatch):
    centers = np.zeros((2, 4), np.float32)
    backend = HNSWCPUBackend(); backend.prepare(centers)
    pending = backend.timings['pending_index_update_s']
    monkeypatch.setattr(fake_hnsw[-1], 'knn_query', lambda *a, **k: (np.array([[-1]]), None))
    with pytest.raises(RuntimeError, match='invalid centroid IDs'):
        backend.assign(centers[:1], centers)
    assert backend.timings['pending_index_update_s'] == pending
    assert backend.timings['query_count'] == 0


def test_query_validation(fake_hnsw):
    centers = np.zeros((2, 4), np.float32)
    backend = HNSWCPUBackend(); backend.prepare(centers)
    for points in (np.zeros((2, 3), np.float32), np.full((2, 4), np.inf, np.float32),
                   np.full((2, 4), 1e30, np.float32)):
        with pytest.raises(ValueError):
            backend.assign(points, centers)
    for effort in (0, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            backend.assign(centers, centers, effort)


real_hnsw = pytest.mark.skipif(importlib.util.find_spec('hnswlib') is None,
                               reason='optional hnswlib is not installed; installation needs approval')


@real_hnsw
@pytest.mark.parametrize('k', [1, 4, 16, 64, 256])
@pytest.mark.parametrize('d', [4, 14])
def test_real_hnsw_quality_reproducibility_and_refresh(k, d):
    rng = np.random.default_rng(19)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    points = rng.normal(size=(129, d)).astype(np.float32)
    exact = assign_exact(points, centers)
    backends = [HNSWCPUBackend(seed=17, query_batch_size=31) for _ in range(2)]
    outputs = []
    for backend in backends:
        backend.prepare(centers)
        result = backend.assign(points, centers, effort=64)
        outputs.append(result)
        delta = points.astype(np.float64) - centers[result.labels].astype(np.float64)
        np.testing.assert_allclose(result.squared_distances, np.sum(delta * delta, axis=1), rtol=1e-14)
        assert np.all(result.squared_distances >= exact.squared_distances - 1e-12)
        recall = np.mean(result.labels == exact.labels)
        gap = (result.squared_distances.sum() - exact.squared_distances.sum()) / max(exact.squared_distances.sum(), 1e-300)
        assert recall >= 0.9 and -1e-12 <= gap <= 0.1
        print(f'K={k}, D={d}: recall={recall:.3f}, objective gap={gap:.6g}')
        moved = centers + np.float32(0.25)
        with pytest.raises(ValueError, match='stale'):
            backend.assign(points, moved)
        backend.refresh(moved)
        updated = backend.assign(points, moved)
        delta = points.astype(np.float64) - moved[updated.labels].astype(np.float64)
        np.testing.assert_allclose(updated.squared_distances, np.sum(delta * delta, axis=1), rtol=1e-14)
    np.testing.assert_array_equal(outputs[0].labels, outputs[1].labels)
    np.testing.assert_array_equal(outputs[0].squared_distances, outputs[1].squared_distances)


@real_hnsw
def test_real_duplicate_ties_and_movement():
    centers = np.array([[0, 0], [0, 0], [2, 0], [2, 0]], np.float32)
    points = np.array([[0, 0], [2, 0], [1, 0]], np.float32)
    backend = HNSWCPUBackend(seed=2); backend.prepare(centers)
    result = backend.assign(points, centers, effort=16)
    assert result.labels[0] == 0 and result.labels[1] == 2
    assert result.labels[2] in (0, 2)  # General equal-distance ANN tie is not globally ordered.
    np.testing.assert_allclose(result.squared_distances, [0, 0, 1])
    moved = centers.copy(); moved[:2] = 20
    backend.refresh(moved)
    np.testing.assert_array_equal(backend.assign(points, moved).labels, [2, 2, 2])


def test_failed_refresh_preserves_previous_index(fake_hnsw, monkeypatch):
    centers = np.zeros((2, 4), np.float32)
    backend = HNSWCPUBackend(); backend.prepare(centers)
    before = backend.timings
    def failed_add(*args, **kwargs):
        raise RuntimeError('injected rebuild failure')
    monkeypatch.setattr(type(fake_hnsw[0]), 'add_items', failed_add)
    with pytest.raises(RuntimeError, match='rebuild failure'):
        backend.refresh(centers + np.float32(1))
    assert backend.timings == before
    np.testing.assert_array_equal(backend.assign(centers, centers).labels, [0, 0])


def test_duplicate_ids_canonicalized_even_when_backend_returns_later_id(fake_hnsw, monkeypatch):
    centers = np.zeros((2, 4), np.float32)
    backend = HNSWCPUBackend(); backend.prepare(centers)
    monkeypatch.setattr(fake_hnsw[-1], 'knn_query', lambda data, **kwargs:
                        (np.ones((len(data), 1), np.int64), np.zeros((len(data), 1))))
    np.testing.assert_array_equal(backend.assign(centers, centers).labels, [0, 0])
