import dataclasses

import pytest

from pcap.adaptive_v2_controller import (
    AdaptiveV2Checkpoint,
    AdaptiveV2Policy,
    initial_v2_state,
    transition_v2,
)


def policy(*, minimum=256, maximum=65536, initial=8192, **kwargs):
    return AdaptiveV2Policy(minimum, maximum, initial, **kwargs)


def checkpoint(q, movement=1.0, labels=1.0):
    return AdaptiveV2Checkpoint(q, movement, labels)


def replay(p, values):
    state = initial_v2_state(p, values[0][0])
    decisions = []
    for objective, movement, labels in values[1:]:
        state, decision = transition_v2(p, state, checkpoint(objective, movement, labels))
        decisions.append(decision)
    return state, decisions


def test_policy_is_immutable_and_has_no_target_or_reference_field():
    p = policy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.maximum = 4
    assert not hasattr(p, "target_sse")
    assert not hasattr(p, "reference_sse")


def test_movement_alone_never_reduces_batch_size():
    p = policy()
    state = initial_v2_state(p, 100.0)
    for _ in range(2):
        state, decision = transition_v2(p, state, checkpoint(100.0, movement=0.0, labels=0.0))
        assert decision.action == "retain"
        assert state.batch_size == p.initial_batch
    state, decision = transition_v2(p, state, checkpoint(100.0, movement=0.0, labels=0.0))
    assert decision.reason == "probe_objective_stagnation"
    assert state.batch_size == 2 * p.initial_batch


def test_only_sustained_probe_progress_and_stability_reduce_batch():
    p = policy()
    state = initial_v2_state(p, 100.0)
    for q in (99.0, 98.0):
        state, decision = transition_v2(p, state, checkpoint(q, movement=0.0001, labels=0.0))
        assert state.batch_size == 8192
        assert decision.action == "retain"
    state, decision = transition_v2(p, state, checkpoint(97.0, movement=0.0001, labels=0.0))
    assert state.batch_size == 4096
    assert decision.reason == "sustained_objective_progress_with_stable_assignments"


def test_probe_deterioration_has_priority_over_stagnation_and_grows_batch():
    p = policy()
    state = initial_v2_state(p, 100.0)
    state, decision = transition_v2(p, state, checkpoint(102.0, movement=0.0, labels=0.0))
    assert decision.deteriorated
    assert decision.reason == "probe_deterioration_restore_safe_batch"
    assert state.batch_size == 16384


def test_oscillation_grows_batch_after_three_significant_alternations():
    p = policy(deterioration=0.1, stagnation_improvement=0.0001,
               progress_improvement=0.005)
    state = initial_v2_state(p, 100.0)
    for q in (99.0, 100.0):
        state, _ = transition_v2(p, state, checkpoint(q))
    state, decision = transition_v2(p, state, checkpoint(99.0))
    assert decision.reason == "probe_objective_oscillation"
    assert state.batch_size == 16384


def test_stagnation_counters_do_not_reset_on_clipped_maximum_noop():
    p = policy(minimum=1024, maximum=8192, initial=8192)
    state = initial_v2_state(p, 100.0)
    for _ in range(5):
        state, decision = transition_v2(p, state, checkpoint(100.0))
    assert state.batch_size == 8192
    assert decision.reason == "probe_objective_stagnation_at_bound"
    assert decision.action == "retain"
    assert not decision.changed
    assert state.stagnation_count == p.required_checkpoints


def test_small_batch_bounds_include_initial_batch():
    p = AdaptiveV2Policy(minimum=256, maximum=4096, initial_batch=512)
    assert (p.minimum, p.initial_batch, p.maximum) == (256, 512, 4096)
    with pytest.raises(ValueError):
        AdaptiveV2Policy(minimum=1024, maximum=4096, initial_batch=512)


def test_zero_probe_objective_is_safe_and_deterministic():
    p = policy()
    state = initial_v2_state(p, 0.0)
    state, decision = transition_v2(p, state, checkpoint(0.0, movement=0.0, labels=0.0))
    assert state.batch_size == p.initial_batch
    assert decision.relative_probe_change == 0.0


def test_deterministic_replay_produces_identical_states_and_decisions():
    p = policy()
    values = [(100.0, 1.0, 1.0), (99.0, .0001, 0.0),
              (98.0, .0001, 0.0), (97.0, .0001, 0.0),
              (97.0, 1.0, 1.0), (97.0, 1.0, 1.0), (97.0, 1.0, 1.0)]
    assert replay(p, values) == replay(p, values)


def test_batch_bounds_and_decision_reasons_are_replayable():
    p = policy(minimum=4096, maximum=16384, initial=8192)
    state = initial_v2_state(p, 100.0)
    for q in (100.0, 100.0, 100.0):
        state, _ = transition_v2(p, state, checkpoint(q))
    assert state.batch_size == 16384
    for q in (163.0, 163.0, 163.0):
        state, decision = transition_v2(p, state, checkpoint(q))
    assert state.batch_size == p.maximum
    assert decision.reason.endswith("at_bound")
    assert p.minimum <= state.batch_size <= p.maximum


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_checkpoint_rejects_invalid_probe_values(bad):
    with pytest.raises((TypeError, ValueError)):
        checkpoint(bad)


def test_invalid_policy_thresholds_are_rejected():
    with pytest.raises(ValueError):
        policy(stagnation_improvement=.01, progress_improvement=.001)
    with pytest.raises(ValueError):
        policy(quiet_label_change=1.1)
