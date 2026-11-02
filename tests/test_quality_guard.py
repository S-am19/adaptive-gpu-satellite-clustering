import dataclasses
import inspect

import numpy as np
import pytest

from pcap import budgeted_minibatch as bm
from pcap.contracts import FitConfig
from pcap.quality_guard_controller import (
    QualityGuardDecision, QualityGuardPolicy, QualityGuardState,
    initial_quality_guard_state, quality_guard_transition,
)


def policy(**kw):
    values = dict(minimum_batch=4, maximum_batch=32, initial_batch=8,
                  required_checkpoints=2, dwell_checkpoints=1)
    values.update(kw)
    return QualityGuardPolicy(**values)


def test_policy_is_pure_and_has_no_evaluation_target_input():
    assert "target_sse" not in inspect.signature(bm.fit_budgeted_minibatch).parameters
    p = policy()
    state = initial_quality_guard_state(p, 10.0)
    next_state, decision = quality_guard_transition(p, state, 10.0, 0.0, 0.0)
    assert state.batch_size == next_state.batch_size == 8
    assert decision.action == "retain"
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.maximum_batch = 64


def test_stagnation_grows_and_progress_requires_stability():
    p = policy(required_checkpoints=2)
    state = initial_quality_guard_state(p, 100.0)
    state, d = quality_guard_transition(p, state, 100.0, 0.0, 0.0)
    assert state.batch_size == 8 and not d.changed
    state, d = quality_guard_transition(p, state, 100.0, 0.0, 0.0)
    assert (state.batch_size, d.reason) == (16, "probe_objective_stagnation")
    p2 = policy(required_checkpoints=2)
    state = initial_quality_guard_state(p2, 100.0)
    state, _ = quality_guard_transition(p2, state, 99.0, 0.0001, 0.0)
    state, d = quality_guard_transition(p2, state, 98.0, 0.0001, 0.0)
    assert state.batch_size == 4 and d.reason == "sustained_progress_and_stability"


def test_deterioration_requests_real_rollback_and_respects_bounds():
    p = policy()
    state = initial_quality_guard_state(p, 10.0)
    state, d = quality_guard_transition(p, state, 11.0, 0.5, 1.0)
    assert d.rollback and d.deteriorated
    assert state.batch_size == 16
    pmax = policy(minimum_batch=8, maximum_batch=8, initial_batch=8)
    state = initial_quality_guard_state(pmax, 10.0)
    state, d = quality_guard_transition(pmax, state, 11.0, 0.5, 1.0)
    assert d.rollback and not d.changed and state.batch_size == 8


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_objectives_are_rejected(value):
    p = policy(); state = initial_quality_guard_state(p, 10.0)
    with pytest.raises(ValueError):
        quality_guard_transition(p, state, value, 0.0, 0.0)


def test_cpu_budget_runner_restores_centroids_and_statistics(monkeypatch):
    rng = np.random.default_rng(9)
    X = np.ascontiguousarray(rng.normal(size=(80, 3)).astype(np.float32))
    C0 = np.ascontiguousarray(X[[0, 20, 40, 60]].copy())
    p = policy(minimum_batch=8, maximum_batch=8, initial_batch=8,
               required_checkpoints=3)
    calls = 0
    original = bm.quality_guard_transition

    def force_one_rollback(policy_arg, state, objective, movement, label_change):
        nonlocal calls
        calls += 1
        if calls == 1:
            after = dataclasses.replace(state, previous_objective=objective,
                                        checkpoints=state.checkpoints + 1)
            decision = QualityGuardDecision(state.batch_size, state.batch_size,
                "retain", "test_forced_deterioration", 1.0, True, True, False)
            return after, decision
        return original(policy_arg, state, objective, movement, label_change)

    monkeypatch.setattr(bm, "quality_guard_transition", force_one_rollback)
    cfg = FitConfig(k=4, seed=9, max_steps=8, tolerance=0.0,
                    batch_size=8, probe_interval=1)
    result, checkpoints, events = bm.fit_budgeted_minibatch(
        X, C0, cfg, device="cpu", observation_budget=32,
        quality_guard_policy=p, evaluation_interval=32, disjoint_probe=True)
    assert events[0]["event"] == "rollback"
    assert result.metadata["cluster_observation_counts"]
    # Subsequent natural probe deterioration may cause additional valid rollbacks.
    assert 0 < sum(result.metadata["cluster_observation_counts"]) < 32
    assert result.metadata["points_processed"] == 32
    assert result.metadata["termination_reason"] == "observation_budget"
    assert checkpoints[-1]["full_exact_sse"] == pytest.approx(result.sse)
    assert result.metadata["quality_guard_uses_evaluation_target"] is False


def test_cpu_budgeted_fixed_and_adaptive_are_reproducible():
    rng = np.random.default_rng(21)
    X = np.ascontiguousarray(rng.normal(size=(96, 4)).astype(np.float32))
    C0 = np.ascontiguousarray(X[[0, 24, 48, 72]].copy())
    cfg = FitConfig(k=4, seed=21, max_steps=20, tolerance=0.0,
                    batch_size=8, probe_interval=2)
    policy_obj = __import__("pcap.adaptive_controller", fromlist=["BatchPolicy"]).BatchPolicy(
        minimum=4, maximum=16, initial_batch=8, quiet_checkpoints=2)
    a = bm.fit_budgeted_minibatch(X, C0, cfg, device="cpu", observation_budget=64,
                                  adaptive=True, batch_policy=policy_obj, evaluation_interval=32)
    b = bm.fit_budgeted_minibatch(X, C0, cfg, device="cpu", observation_budget=64,
                                  adaptive=True, batch_policy=policy_obj, evaluation_interval=32)
    assert np.array_equal(a[0].centroids, b[0].centroids)
    assert np.array_equal(a[0].labels, b[0].labels)
    assert [x["actual_batch_size"] for x in a[0].history] == [x["actual_batch_size"] for x in b[0].history]
    assert a[0].metadata["points_processed"] == 64


def test_cuda_quality_guard_restores_gpu_centroids_and_statistics(monkeypatch):
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        pytest.skip("real CUDA GPU required")
    rng = np.random.default_rng(29)
    X = np.ascontiguousarray(rng.normal(size=(2048, 4)).astype(np.float32))
    C0 = np.ascontiguousarray(X[[0, 512, 1024, 1536]].copy())
    p = policy(minimum_batch=8, maximum_batch=8, initial_batch=8,
               required_checkpoints=3)
    original = bm.quality_guard_transition
    calls = 0

    def force_first_rollback(policy_arg, state, objective, movement, label_change):
        nonlocal calls
        calls += 1
        if calls == 1:
            after = dataclasses.replace(state, previous_objective=objective,
                                        checkpoints=state.checkpoints + 1)
            return after, QualityGuardDecision(state.batch_size, state.batch_size,
                "retain", "test_forced_deterioration", 1.0, True, True, False)
        return original(policy_arg, state, objective, movement, label_change)

    monkeypatch.setattr(bm, "quality_guard_transition", force_first_rollback)
    cfg = FitConfig(k=4, seed=29, max_steps=8, tolerance=0.0,
                    batch_size=8, probe_interval=1)
    result, checkpoints, events = bm.fit_budgeted_minibatch(
        X, C0, cfg, device="cuda", observation_budget=32,
        quality_guard_policy=p, evaluation_interval=32, disjoint_probe=True)
    assert any(event["event"] == "rollback" for event in events)
    assert 0 < sum(result.metadata["cluster_observation_counts"]) < 32
    assert result.labels.dtype == np.int32
    assert result.sse == pytest.approx(checkpoints[-1]["full_exact_sse"], rel=1e-10)
    assert result.metadata["execution_device"] == "cuda"


def test_budgeted_fixed_cuda_matches_production_g1_sampling_and_update():
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        pytest.skip("real CUDA GPU required")
    from pcap.minibatch_cuda import fit_minibatch_cuda
    rng = np.random.default_rng(37)
    X = np.ascontiguousarray(rng.normal(size=(2048, 4)).astype(np.float32))
    C0 = np.ascontiguousarray(X[[0, 512, 1024, 1536]].copy())
    cfg = FitConfig(k=4, seed=37, max_steps=3, tolerance=0.0,
                    batch_size=64, probe_interval=10)
    production = fit_minibatch_cuda(X, C0, cfg)
    replay, _, _ = bm.fit_budgeted_minibatch(
        X, C0, cfg, device="cuda", observation_budget=192,
        evaluation_interval=192, disjoint_probe=False)
    np.testing.assert_allclose(replay.centroids, production.centroids, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(replay.labels, production.labels)
    assert replay.sse == pytest.approx(production.sse, rel=1e-9)
    assert replay.metadata["points_processed"] == production.metadata["points_processed"] == 192
