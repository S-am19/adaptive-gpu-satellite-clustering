"""Fixed Mini-Batch correctness tests; no throughput benchmarks."""

import math

import numpy as np
import pytest

from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig
from pcap.kmeans_cpu import fit_cpu
from pcap.minibatch import fit_minibatch_cpu, _sample_batch_indices


def config(k, b, steps=7, seed=0, interval=100, tolerance=0):
    return FitConfig(k=k, seed=seed, max_steps=steps, tolerance=tolerance,
                     batch_size=b, probe_interval=interval)


def independent_reference(points, initial, cfg):
    rng = np.random.default_rng(cfg.seed)
    centers = initial.copy()
    sums = np.zeros(initial.shape, np.float64)
    counts = np.zeros(len(initial), np.int64)
    for _ in range(cfg.max_steps):
        indices = np.arange(len(points)) if cfg.batch_size == len(points) else rng.choice(
            len(points), size=cfg.batch_size, replace=False)
        batch = points[indices].astype(np.float64)
        distances = np.sum((batch[:, None, :] - centers.astype(np.float64)[None, :, :])**2, axis=2)
        labels = np.argmin(distances, axis=1)
        for j in range(len(centers)):
            selected = batch[labels == j]
            if len(selected):
                sums[j] += selected.sum(axis=0)
                counts[j] += len(selected)
                centers[j] = sums[j]/counts[j]
    return centers, counts


@pytest.mark.parametrize('n,d,k,b', [(19, 1, 1, 1), (37, 4, 3, 7),
                                    (67, 14, 4, 13), (15, 3, 3, 15)])
def test_multistep_matches_independent_numpy(n, d, k, b):
    points = np.random.default_rng(12).normal(size=(n, d)).astype(np.float32)
    initial = points[:k].copy()
    cfg = config(k, b, steps=8, seed=17)
    saved_points, saved_initial = points.copy(), initial.copy()
    result = fit_minibatch_cpu(points, initial, cfg)
    centers, counts = independent_reference(points, initial, cfg)
    np.testing.assert_allclose(result.centroids, centers, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], counts)
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(initial, saved_initial)
    assert not np.shares_memory(result.centroids, initial)
    exact = assign_exact(points, result.centroids)
    np.testing.assert_array_equal(result.labels, exact.labels)
    assert result.sse == pytest.approx(math.fsum(exact.squared_distances), rel=1e-14)
    assert result.centroids.dtype == np.float32 and result.centroids.flags.c_contiguous
    assert result.labels.dtype == np.int32
    assert result.metadata['steps'] == cfg.max_steps
    assert result.metadata['points_processed'] == cfg.max_steps*b
    assert result.metadata['assignment_points_processed'] == cfg.max_steps*b + min(n, 1024) + n
    assert 'sse' not in result.history[0]
    assert all(row['cumulative_observations'] == row['step']*b for row in result.history)


def test_known_one_step_and_full_batch_cpu_equivalence(two_groups):
    points, initial = two_groups
    result = fit_minibatch_cpu(points, initial, config(2, 4, steps=1))
    np.testing.assert_array_equal(result.centroids, [[0, 1], [10, 11]])
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], [2, 2])
    assert result.sse == 4
    cpu = fit_cpu(points, initial, FitConfig(2, 0, 1, 0))
    np.testing.assert_array_equal(result.centroids, cpu.centroids)
    np.testing.assert_array_equal(result.labels, cpu.labels)


def test_cumulative_counts_not_reset_or_seeded_from_C0():
    points = np.array([[-10], [-9], [-8], [1], [2], [3]], np.float32)
    initial = np.array([[-10], [-9]], np.float32)
    result = fit_minibatch_cpu(points, initial, config(2, 6, steps=2))
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], [4, 8])
    np.testing.assert_array_equal(result.centroids, [[-9.25], [-0.625]])


def test_no_batch_points_retains_centroid_and_ties_choose_lowest():
    points = np.ones((10, 3), np.float32)
    initial = np.array([[1, 1, 1], [1, 1, 1], [100, 100, 100]], np.float32)
    result = fit_minibatch_cpu(points, initial, config(3, 3, steps=5))
    np.testing.assert_array_equal(result.centroids, initial)
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], [15, 0, 0])
    np.testing.assert_array_equal(result.labels, np.zeros(10, np.int32))


def test_seeded_reproducibility_and_batch_policy():
    points = np.random.default_rng(9).normal(size=(80, 4)).astype(np.float32)
    cfg = config(3, 7, seed=29)
    runs = [fit_minibatch_cpu(points, points[:3].copy(), cfg) for _ in range(2)]
    np.testing.assert_array_equal(runs[0].centroids, runs[1].centroids)
    np.testing.assert_array_equal(runs[0].labels, runs[1].labels)
    assert runs[0].history == runs[1].history and runs[0].sse == runs[1].sse
    assert runs[0].metadata == runs[1].metadata
    rng = np.random.default_rng(5)
    batches = [_sample_batch_indices(rng, 20, 7) for _ in range(5)]
    assert all(len(np.unique(batch)) == 7 for batch in batches)
    assert any(np.intersect1d(batches[0], batch).size for batch in batches[1:])
    np.testing.assert_array_equal(_sample_batch_indices(rng, 20, 20), np.arange(20))


def test_stopping_requires_three_checkpoint_checks():
    points = np.ones((20, 2), np.float32)
    result = fit_minibatch_cpu(points, points[:1].copy(), config(1, 4, steps=20, interval=2))
    assert result.metadata['steps'] == 6
    assert result.metadata['termination_reason'] == 'tolerance'
    assert result.metadata['convergence_satisfied'] is True
    assert result.metadata['probe_checks'] == 3
    assert [row['step'] for row in result.history if row['convergence_checked']] == [2, 4, 6]
    assert result.history[-1]['convergence_satisfied']
    assert result.sse == 0
    capped = fit_minibatch_cpu(points, points[:1].copy(), config(1, 4, steps=5, interval=2))
    assert capped.metadata['termination_reason'] == 'max_steps'
    assert capped.metadata['convergence_satisfied'] is False


def test_batch_sse_is_not_final_sse_and_full_objective_can_increase(monkeypatch):
    import pcap.minibatch as module
    points = np.array([[0], [10]], np.float32)
    initial = np.array([[5]], np.float32)
    monkeypatch.setattr(module, '_sample_batch_indices', lambda *args: np.array([0]))
    result = fit_minibatch_cpu(points, initial, config(1, 1, steps=1))
    assert result.history[0]['batch_sse_before_update'] == 25
    assert result.sse == 100 > assign_exact(points, initial).squared_distances.sum()
    result.history[0]['batch_sse_before_update'] = -123
    assert result.sse == 100  # RunResult reads only the exact final objective.


def test_final_full_assignment_exactly_once(monkeypatch, two_groups):
    import pcap.minibatch as module
    original = module.assign_exact
    calls = []
    def final(points, centroids):
        calls.append(len(points))
        return original(points, centroids)
    monkeypatch.setattr(module, 'assign_exact', final)
    result = fit_minibatch_cpu(*two_groups, config(2, 2, steps=3, interval=1))
    assert calls == [4]
    t = result.timings
    assert all(np.isfinite(v) and v >= 0 for v in t.values())
    assert t['final_assignment_s'] <= t['compute_s'] <= t['wall_s']
    assert sum(t[name] for name in ('sampling_s', 'batch_assignment_s', 'accumulation_update_s',
                                   'monitoring_s', 'final_assignment_s')) <= t['compute_s']


@pytest.mark.parametrize('scale', [1e-20, 1e20, 1e30])
def test_finite_large_and_small_values(scale):
    points = np.array([[-1, 2], [1, -2], [-0.5, 1], [0.5, -1]], np.float32)*np.float32(scale)
    initial = points[[0, 1]].copy()
    cfg = config(2, 3, steps=5)
    result = fit_minibatch_cpu(points, initial, cfg)
    reference, counts = independent_reference(points, initial, cfg)
    np.testing.assert_allclose(result.centroids, reference, rtol=2e-6, atol=abs(scale)*1e-7)
    assert np.isfinite(result.sse)
    exact = assign_exact(points, result.centroids)
    assert result.sse == pytest.approx(math.fsum(exact.squared_distances), rel=1e-14)


@pytest.mark.parametrize('bad', [np.zeros((4, 2), np.float64), np.zeros((0, 2), np.float32),
                                np.full((4, 2), np.nan, np.float32),
                                np.full((4, 2), np.inf, np.float32),
                                np.zeros((4, 4), np.float32)[:, ::2]])
def test_invalid_points(bad):
    with pytest.raises((TypeError, ValueError)):
        fit_minibatch_cpu(bad, np.zeros((1, 2), np.float32), config(1, 1))


@pytest.mark.parametrize('kind', ['config', 'k', 'dimension', 'missing_batch', 'large_batch',
                                 'ann', 'count_overflow', 'centroid_dtype', 'centroid_nan'])
def test_invalid_configuration_and_centroids(kind):
    points = np.zeros((4, 2), np.float32); initial = points[:1].copy(); cfg = config(1, 2)
    if kind == 'config': cfg = {}
    if kind == 'k': cfg = config(2, 2)
    if kind == 'dimension': initial = np.zeros((1, 3), np.float32)
    if kind == 'missing_batch': cfg = FitConfig(1, 0, 2, 0)
    if kind == 'large_batch': cfg = config(1, 5)
    if kind == 'ann': cfg = FitConfig(1, 0, 2, 0, batch_size=2, ann_effort=4)
    if kind == 'count_overflow': cfg = config(1, 2, steps=np.iinfo(np.int64).max)
    if kind == 'centroid_dtype': initial = initial.astype(np.float64)
    if kind == 'centroid_nan': initial[:] = np.nan
    with pytest.raises((TypeError, ValueError)):
        fit_minibatch_cpu(points, initial, cfg)


def test_probe_schedule_does_not_change_batch_sampling():
    points = np.random.default_rng(52).normal(size=(51, 4)).astype(np.float32)
    initial = points[:3].copy()
    sparse = fit_minibatch_cpu(points, initial, config(3, 8, steps=5, seed=13, interval=100))
    frequent = fit_minibatch_cpu(points, initial, config(3, 8, steps=5, seed=13, interval=1))
    assert sparse.metadata['steps'] == frequent.metadata['steps'] == 5
    np.testing.assert_array_equal(sparse.centroids, frequent.centroids)
    np.testing.assert_array_equal(sparse.metadata['cluster_observation_counts'],
                                  frequent.metadata['cluster_observation_counts'])
    other = fit_minibatch_cpu(points, initial, config(3, 8, steps=5, seed=14))
    assert not np.array_equal(sparse.centroids, other.centroids)


@pytest.mark.parametrize('steps,batch', [
    (np.int64(2**62), np.int64(4)),
    (np.int64(np.iinfo(np.int64).max), np.int64(2)),
    (np.uint64(2**63), np.uint64(4)),
])
def test_numpy_integer_observation_capacity_does_not_wrap(steps, batch):
    points = np.zeros((4, 2), np.float32)
    cfg = config(1, batch, steps=steps)
    # A NumPy overflow must not happen even when configured to raise.
    with np.errstate(over='raise'):
        with pytest.raises(ValueError, match='exceeds int64 capacity'):
            fit_minibatch_cpu(points, points[:1].copy(), cfg)


def test_exact_int64_observation_capacity_is_accepted():
    points = np.ones((4, 2), np.float32)
    cfg = config(1, np.int64(1), steps=np.int64(np.iinfo(np.int64).max), interval=1)
    # Exact-boundary configuration is valid; the stable fixture stops at 3.
    result = fit_minibatch_cpu(points, points[:1].copy(), cfg)
    assert result.metadata['steps'] == 3
    assert result.metadata['termination_reason'] == 'tolerance'
    assert result.metadata['points_processed'] == 3


@pytest.mark.parametrize('limit,expected_steps,reason', [(5, 5, 'max_steps'),
                                                       (6, 6, 'tolerance'),
                                                       (10, 6, 'tolerance')])
def test_unstable_checkpoint_resets_counter_and_cap_boundary(monkeypatch, limit, expected_steps, reason):
    import pcap.minibatch as module
    points = np.array([[0], [4], [12]], np.float32)
    # Running means: 0, 0, 4, 4, 4, 4. The third checkpoint interrupts
    # two stable checks; three NEW stable checks are needed afterward.
    draws = iter([0, 0, 2, 1, 1, 1])
    monkeypatch.setattr(module, '_sample_batch_indices', lambda *args:
                        np.array([next(draws)]))
    result = fit_minibatch_cpu(points, points[:1].copy(),
                              config(1, 1, steps=limit, interval=1))
    counters = [1, 2, 0, 1, 2, 3][:expected_steps]
    assert [row['stable_checks'] for row in result.history] == counters
    assert all(row['convergence_checked'] for row in result.history)
    assert result.history[2]['checkpoint_displacement'] == 4
    assert result.metadata['steps'] == expected_steps
    assert result.metadata['probe_checks'] == expected_steps
    assert result.metadata['termination_reason'] == reason
    assert result.metadata['convergence_satisfied'] == (reason == 'tolerance')
    assert result.history[-1]['convergence_satisfied'] == (reason == 'tolerance')
    assert result.metadata['cluster_observation_counts'] == [expected_steps]
    assert result.history[-1]['batch_sse_before_update'] == 0
    assert result.sse == 80
    exact = assign_exact(points, result.centroids)
    np.testing.assert_array_equal(result.labels, exact.labels)
