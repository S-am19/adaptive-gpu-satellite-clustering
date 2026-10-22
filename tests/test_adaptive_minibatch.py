"""Adaptive execution arithmetic, controller wiring and confirmation regressions."""

from dataclasses import replace
import math

import numpy as np
import pytest

from pcap.adaptive_controller import BatchPolicy, Checkpoint, transition
from pcap.adaptive_minibatch import fit_adaptive_minibatch_cpu
from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig


def cfg(k=1, batch=4, steps=12, seed=0, interval=1, tolerance=0):
    return FitConfig(k, seed, steps, tolerance, batch_size=batch, probe_interval=interval)


def reference(X, C0, seed, sizes):
    rng = np.random.default_rng(seed)
    centers = C0.copy()
    sums = np.zeros(C0.shape, np.float64)
    counts = np.zeros(len(C0), np.int64)
    for b in sizes:
        ix = np.arange(len(X)) if b == len(X) else rng.choice(len(X), b, replace=False)
        batch = X[ix].astype(np.float64)
        distances = ((batch[:, None] - centers.astype(np.float64)[None])**2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for j in range(len(C0)):
            chosen = batch[labels == j]
            if len(chosen):
                sums[j] += chosen.sum(axis=0)
                counts[j] += len(chosen)
                centers[j] = sums[j]/counts[j]
    return centers, counts


def assert_final(X, result):
    distances = ((X.astype(np.float64)[:, None] - result.centroids.astype(np.float64)[None])**2).sum(axis=2)
    labels = distances.argmin(axis=1).astype(np.int32)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.sse == pytest.approx(math.fsum(distances[np.arange(len(X)), labels]), rel=1e-14)
    assert result.centroids.dtype == np.float32 and result.centroids.flags.c_contiguous
    assert result.labels.dtype == np.int32 and result.labels.flags.c_contiguous


@pytest.mark.parametrize('n,d,k,b', [(31, 4, 3, 4), (71, 14, 4, 9), (17, 1, 1, 3)])
def test_multistep_reference_and_actual_controller_replay(n, d, k, b):
    X = np.random.default_rng(15).normal(size=(n, d)).astype(np.float32)
    C0 = X[:k].copy()
    original = X.copy(), C0.copy()
    p = BatchPolicy.for_run(n, b, minimum=1, maximum=n)
    result = fit_adaptive_minibatch_cpu(X, C0, cfg(k, b, 12, seed=17), policy=p)
    centers, counts = reference(X, C0, 17, [r['batch_size'] for r in result.history])
    np.testing.assert_allclose(result.centroids, centers, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], counts)
    np.testing.assert_array_equal(X, original[0]); np.testing.assert_array_equal(C0, original[1])
    total = 0
    for row in result.history:
        total += row['batch_size']
        assert row['points_processed'] == row['cumulative_observations'] == total
        if row['controller_decision'] is not None:
            from pcap.adaptive_controller import ControllerState
            state = ControllerState(**row['controller_state_before'])
            updated, decision = transition(p, state, Checkpoint(row['probe_mean_squared_distance'],
                row['normalized_checkpoint_displacement'], row['probe_label_change_fraction']))
            assert row['controller_state_after'] == vars(updated)
            assert row['controller_decision'] == {**vars(decision), 'changed': decision.changed}
    assert result.metadata['points_processed'] == total == int(counts.sum())
    assert_final(X, result)


def test_genuine_decreases_and_persistent_rng_counts():
    X = np.ones((64, 4), np.float32)
    p = BatchPolicy.for_run(64, 16, minimum=2, maximum=32)
    result = fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=16, steps=14), policy=p)
    decisions = [r['controller_decision'] for r in result.history if r['controller_decision']]
    assert [d['next_batch'] for d in decisions if d['changed']] == [8, 4, 2]
    assert result.metadata['confirmation_start_step'] == 12
    assert result.metadata['confirmation_checks'] == 2
    assert not result.metadata['convergence_satisfied']
    assert result.metadata['termination_reason'] == 'max_steps'
    assert [r['batch_size'] for r in result.history] == [16]*3+[8]*3+[4]*3+[2]*3+[32]*2
    assert result.metadata['cluster_observation_counts'] == [154]
    assert_final(X, result)


def test_deterioration_increases_and_clipped_noops(monkeypatch):
    import pcap.adaptive_minibatch as module
    X = np.array([[0]]*4+[[10]]*4, np.float32)
    monkeypatch.setattr(module, '_sample_batch_indices', lambda rng, n, b: np.arange(b))
    p = BatchPolicy.for_run(8, 1, minimum=1, maximum=4)
    result = module.fit_adaptive_minibatch_cpu(X, np.array([[5]], np.float32), cfg(batch=1, steps=6), policy=p)
    decisions = [r['controller_decision'] for r in result.history]
    assert [d['next_batch'] for d in decisions] == [2, 4, 4, 4, 4, 4]
    assert all(d['reason'] == 'deterioration_at_bound' and not d['changed'] for d in decisions[2:])
    assert result.metadata['cluster_observation_counts'] == [19]
    assert not result.metadata['convergence_satisfied']
    assert result.metadata['confirmation_start_step'] is None
    assert_final(X, result)


@pytest.mark.parametrize('limit,steps,reason,checks', [(5, 5, 'max_steps', 2),
                                                     (6, 6, 'tolerance', 3),
                                                     (9, 6, 'tolerance', 3)])
def test_confirmation_cap_boundary_and_frozen_transition(monkeypatch, limit, steps, reason, checks):
    import pcap.adaptive_minibatch as module
    real_transition = module.transition
    calls = []
    def tracked(*args):
        calls.append(args)
        return real_transition(*args)
    monkeypatch.setattr(module, 'transition', tracked)
    X = np.ones((3, 2), np.float32)
    p = BatchPolicy.for_run(3, 1, minimum=1, maximum=1)
    result = module.fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=1, steps=limit), policy=p)
    assert result.metadata['steps'] == steps
    assert result.metadata['termination_reason'] == reason
    assert result.metadata['convergence_satisfied'] == (reason == 'tolerance')
    assert result.metadata['confirmation_checks'] == checks
    assert result.metadata['confirmation_start_step'] == 3
    assert len(calls) == result.metadata['controller_calls'] == 3
    assert all(r['controller_decision'] is None and r['adaptation_frozen'] for r in result.history[3:])
    assert result.metadata['points_processed'] == steps


def test_confirmation_instability_resets_and_stays_frozen(monkeypatch):
    import pcap.adaptive_minibatch as module
    X = np.array([[0], [12]], np.float32)
    monkeypatch.setattr(module, '_sample_batch_indices', lambda rng, n, b: np.arange(b))
    p = BatchPolicy.for_run(2, 1, minimum=1, maximum=2)
    result = module.fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=1, steps=7), policy=p)
    assert result.metadata['confirmation_start_step'] == 3
    assert result.history[3]['checkpoint_displacement'] > 0
    assert all(r['confirmation_checks'] == 0 for r in result.history[3:])
    assert all(r['batch_size'] == 2 and r['controller_decision'] is None for r in result.history[3:])
    assert result.metadata['termination_reason'] == 'max_steps'
    assert result.metadata['controller_calls'] == 3
    assert_final(X, result)


def test_reproducibility_and_probe_rng_independence():
    X = np.random.default_rng(19).normal(size=(1300, 4)).astype(np.float32)
    C0 = X[:3].copy()
    p = BatchPolicy.for_run(len(X), 16, minimum=16, maximum=16)
    config = cfg(3, 16, 5, seed=41, interval=1)
    first = fit_adaptive_minibatch_cpu(X, C0, config, policy=p)
    second = fit_adaptive_minibatch_cpu(X, C0, config, policy=p)
    assert first.history == second.history and first.metadata == second.metadata
    np.testing.assert_array_equal(first.centroids, second.centroids)
    sparse = fit_adaptive_minibatch_cpu(X, C0, replace(config, probe_interval=100), policy=p)
    np.testing.assert_array_equal(first.centroids, sparse.centroids)
    np.testing.assert_array_equal(first.metadata['probe_indices'], sparse.metadata['probe_indices'])
    assert_final(X, first)


def test_empty_clusters_and_default_small_dataset():
    X = np.ones((4, 2), np.float32)
    C0 = np.array([[1, 1], [1, 1], [100, 100]], np.float32)
    result = fit_adaptive_minibatch_cpu(X, C0, cfg(3, 1, 6))
    assert result.metadata['policy']['minimum'] == 1
    assert result.metadata['policy']['maximum'] == 4
    np.testing.assert_array_equal(result.centroids, C0)
    assert result.metadata['cluster_observation_counts'][1:] == [0, 0]
    assert_final(X, result)


@pytest.mark.parametrize('scale', [1e-20, 1e20, 1e30])
def test_numeric_scales(scale):
    X = np.array([[-1, 2], [1, -2], [-.5, 1], [.5, -1]], np.float32)*np.float32(scale)
    result = fit_adaptive_minibatch_cpu(X, X[:2].copy(), cfg(2, 2, 7))
    assert np.isfinite(result.metadata['probe_scale'])
    assert_final(X, result)


def test_final_exact_once_and_timing_scopes(monkeypatch):
    import pcap.adaptive_minibatch as module
    original = module.assign_exact
    calls = []
    def tracked(*args):
        calls.append(args[0].shape)
        return original(*args)
    monkeypatch.setattr(module, 'assign_exact', tracked)
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    result = module.fit_adaptive_minibatch_cpu(X, X[:2].copy(), cfg(2, 4, 7))
    assert calls == [X.shape]
    t = result.timings
    assert all(np.isfinite(v) and v >= 0 for v in t.values())
    assert t['final_assignment_s'] <= t['compute_s'] <= t['wall_s']
    assert sum(t[k] for k in t if k not in ('wall_s', 'compute_s')) <= t['compute_s']
    assert result.metadata['assignment_points_processed'] == result.metadata['points_processed'] + (result.metadata['probe_checks']+1)*20 + 20
    assert all('sse' not in r for r in result.history)


@pytest.mark.parametrize('kind', ['dtype', 'nan', 'dimension', 'k', 'config', 'missing_batch',
                                 'large_batch', 'ann', 'policy_type', 'policy_initial', 'policy_max'])
def test_invalid_inputs(kind):
    X = np.zeros((4, 2), np.float32); C0 = X[:1].copy(); config = cfg(batch=1); p = None
    if kind == 'dtype': X = X.astype(np.float64)
    if kind == 'nan': X[0, 0] = np.nan
    if kind == 'dimension': C0 = np.zeros((1, 3), np.float32)
    if kind == 'k': config = cfg(k=2, batch=1)
    if kind == 'config': config = {}
    if kind == 'missing_batch': config = replace(config, batch_size=None)
    if kind == 'large_batch': config = replace(config, batch_size=5)
    if kind == 'ann': config = replace(config, ann_effort=4)
    if kind == 'policy_type': p = {}
    if kind == 'policy_initial': p = BatchPolicy.for_run(4, 2)
    if kind == 'policy_max': p = BatchPolicy(1, 5, 1)
    with pytest.raises((TypeError, ValueError)):
        fit_adaptive_minibatch_cpu(X, C0, config, policy=p)


@pytest.mark.parametrize('steps', [np.int64(2**62), np.uint64(2**63)])
def test_overflow_uses_python_integers_and_maximum_not_initial(steps):
    X = np.zeros((4, 1), np.float32)
    with np.errstate(over='raise'):
        with pytest.raises(ValueError, match='int64 capacity'):
            fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=np.int64(1), steps=steps))


def test_exact_capacity_valid_and_early_stop():
    X = np.ones((1, 1), np.float32)
    result = fit_adaptive_minibatch_cpu(X, X.copy(), cfg(batch=np.int64(1), steps=np.int64(np.iinfo(np.int64).max)))
    assert result.metadata['steps'] == 6
    assert result.metadata['termination_reason'] == 'tolerance'


@pytest.mark.parametrize('limit,reason', [(14, 'max_steps'), (15, 'tolerance')])
def test_confirmation_after_actual_size_changes_at_cap(limit, reason):
    X = np.ones((64, 1), np.float32)
    p = BatchPolicy.for_run(64, 16, minimum=2, maximum=32)
    result = fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=16, steps=limit), policy=p)
    assert result.metadata['steps'] == limit
    assert result.metadata['termination_reason'] == reason
    assert result.metadata['confirmation_checks'] == limit-12
    assert all(r['batch_size'] == 32 for r in result.history[12:])


def test_checkpoint_cadence_and_confirmation_quality_gate(monkeypatch):
    import pcap.adaptive_minibatch as module
    X = np.array([[0], [12]], np.float32)
    monkeypatch.setattr(module, '_sample_batch_indices', lambda rng, n, b: np.arange(b))
    # C0=6 has a better probe objective than the early biased cumulative means.
    # With generous movement tolerance, check checkpoint cadence and the
    # recorded confirmation gate against its independent formula.
    p = BatchPolicy.for_run(2, 1, minimum=1, maximum=2, deterioration=100)
    result = module.fit_adaptive_minibatch_cpu(X, np.array([[6]], np.float32),
                                             cfg(batch=1, steps=12, interval=2, tolerance=100), policy=p)
    assert [r['step'] for r in result.history if r['convergence_checked']] == [2, 4, 6, 8, 10, 12]
    assert result.metadata['confirmation_start_step'] == 6
    assert result.metadata['controller_calls'] == 3
    assert all(r['controller_decision'] is None for r in result.history if r['step'] % 2)
    for row in result.history[6:]:
        if row['convergence_checked']:
            q0 = result.metadata['confirmation_objective']
            assert row['confirmation_quality_satisfied'] == (row['probe_mean_squared_distance'] <= q0+p.deterioration*max(q0,p.objective_floor))


def test_quality_deterioration_resets_confirmation_without_unfreezing(monkeypatch):
    import pcap.adaptive_minibatch as module
    original = module.ExactCPUBackend.assign
    probe_calls = []
    # Inject monitoring objectives only to isolate the stopping state machine.
    # Updates and the final full-data oracle remain genuine exact assignments.
    def controlled_probe(self, points, centroids, effort=None):
        result = original(self, points, centroids, effort)
        if len(points) == 8:
            probe_calls.append(1)
            # Initial + 3 candidate probes, then one good confirmation,
            # a deteriorated confirmation, followed by three good ones.
            if len(probe_calls) == 6:
                result.squared_distances = np.ones(8, np.float64)
        return result
    monkeypatch.setattr(module.ExactCPUBackend, 'assign', controlled_probe)
    X = np.ones((8, 1), np.float32)
    p = BatchPolicy.for_run(8, 1, minimum=1, maximum=2)
    result = module.fit_adaptive_minibatch_cpu(X, X[:1].copy(), cfg(batch=1, steps=9), policy=p)
    assert result.metadata['confirmation_start_step'] == 3
    assert [r['confirmation_checks'] for r in result.history[3:]] == [1, 0, 1, 2, 3]
    assert not result.history[4]['confirmation_quality_satisfied']
    assert result.metadata['controller_calls'] == 3
    assert result.metadata['steps'] == 8 and result.metadata['termination_reason'] == 'tolerance'
    assert_final(X, result)


def test_independent_multicluster_reference_across_multiple_real_changes():
    X = np.random.default_rng(27).normal(size=(101, 4)).astype(np.float32)
    C0 = X[[0, 25, 50]].copy()
    # Permissive test thresholds force real quiet decisions, while random
    # nonzero centroid movement prevents the separate zero-tolerance stop.
    p = BatchPolicy.for_run(len(X), 16, minimum=2, maximum=32,
                            quiet_movement=1e20, quiet_label_change=1,
                            quiet_objective_change=1e20, oscillation_change=1e20,
                            deterioration=1e20)
    result = fit_adaptive_minibatch_cpu(X, C0, cfg(3, 16, 12, seed=9), policy=p)
    sizes = [r['batch_size'] for r in result.history]
    assert sizes == [16]*3 + [8]*3 + [4]*3 + [2]*3
    centers, counts = reference(X, C0, 9, sizes)
    np.testing.assert_allclose(result.centroids, centers, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result.metadata['cluster_observation_counts'], counts)
    assert sum(sizes) == int(counts.sum()) == result.metadata['points_processed']
    assert_final(X, result)
