"""Pure controller correctness and replay; no clustering or timing experiments."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pcap.adaptive_controller import (
    BatchPolicy, Checkpoint, ControllerState, initial_state, transition,
)


def policy(**kw):
    return BatchPolicy.for_run(1000, 64, minimum=8, maximum=256, **kw)


def replay(p, observations, objective=100):
    state = initial_state(p, objective)
    result = []
    for observation in observations:
        state, decision = transition(p, state, observation)
        result.append((state, decision))
    return result


@pytest.mark.parametrize('n,b,lo,hi', [(1000, 32, 32, 1000), (100, 1, 1, 100),
                                      (100, 100, 100, 100), (100000, 70000, 256, 70000),
                                      (100000, 8192, 256, 65536)])
def test_default_bounds_include_initial(n, b, lo, hi):
    p = BatchPolicy.for_run(n, b)
    assert (p.minimum, p.maximum) == (lo, hi)
    assert p.minimum <= p.initial_batch <= p.maximum <= n


@pytest.mark.parametrize('kwargs', [dict(n=0, initial_batch=1), dict(n=10, initial_batch=11),
                                   dict(n=10, initial_batch=2, minimum=3),
                                   dict(n=10, initial_batch=2, maximum=11),
                                   dict(n=10, initial_batch=2, maximum=1),
                                   dict(n=True, initial_batch=1)])
def test_invalid_bounds(kwargs):
    with pytest.raises((TypeError, ValueError)):
        BatchPolicy.for_run(**kwargs)


@pytest.mark.parametrize('kw', [dict(deterioration=-1), dict(quiet_movement=np.nan),
                               dict(objective_floor=0), dict(objective_floor=np.inf),
                               dict(quiet_label_change=1.1), dict(quiet_checkpoints=0),
                               dict(quiet_checkpoints=True), dict(oscillation_change=-1)])
def test_invalid_thresholds(kw):
    with pytest.raises((TypeError, ValueError)):
        policy(**kw)


@pytest.mark.parametrize('args', [(-1, 0, 0), (np.inf, 0, 0), (1, -1, 0),
                                 (1, 0, 1.1), (1, np.nan, 0), (True, 0, 0)])
def test_invalid_checkpoint(args):
    with pytest.raises((ValueError, TypeError)):
        Checkpoint(*args)


def test_deterioration_threshold_strict_and_default_one_percent():
    p = policy()
    state = initial_state(p, 100)
    same, boundary = transition(p, state, Checkpoint(101, 1, 1))
    assert boundary.action == 'retain' and not boundary.deteriorated
    higher, over = transition(p, state, Checkpoint(np.nextafter(101., np.inf), 1, 1))
    assert higher.batch_size == 128 and over.reason == 'deterioration'
    assert over.changed
    assert state == initial_state(p, 100)  # No mutation.


def test_quiet_thresholds_inclusive_and_three_checkpoints():
    p = policy(quiet_movement=.25, quiet_label_change=.25,
               quiet_objective_change=.25, deterioration=1)
    rows = replay(p, [Checkpoint(100, .25, .25)] * 3)
    assert [r[0].batch_size for r in rows] == [64, 64, 32]
    assert [r[0].quiet_count for r in rows] == [1, 2, 0]
    assert rows[-1][1].reason == 'quiet'
    # Relative objective equality is also inclusive (exact binary fractions).
    _, decision = transition(p, initial_state(p, 100), Checkpoint(125, .25, .25))
    assert decision.quiet
    for cp in (Checkpoint(100, .25001, .25), Checkpoint(100, .25, .25001),
               Checkpoint(125.001, .25, .25)):
        _, decision = transition(p, initial_state(p, 100), cp)
        assert not decision.quiet


def test_unstable_checkpoint_resets_quiet_streak():
    p = policy()
    observations = [Checkpoint(100, 0, 0)] * 2 + [Checkpoint(100, .1, 0)] + [Checkpoint(100, 0, 0)] * 3
    rows = replay(p, observations)
    assert [s.quiet_count for s, _ in rows] == [1, 2, 0, 1, 2, 0]
    assert [s.batch_size for s, _ in rows] == [64] * 5 + [32]


def test_significant_three_change_oscillation_and_priority():
    p = policy(deterioration=1)
    rows = replay(p, [Checkpoint(q, 1, 1) for q in (101, 100, 101)])
    assert [d.action for _, d in rows] == ['retain', 'retain', 'increase']
    assert rows[-1][1].reason == 'oscillation'
    assert rows[-1][0].recent_changes == ()
    # Both oscillation and deterioration hold: deterioration must win.
    rows = replay(policy(), [Checkpoint(q, 1, 1) for q in (101, 100, 102)])
    assert rows[-1][1].oscillating and rows[-1][1].deteriorated
    assert rows[-1][1].reason == 'deterioration'


def test_oscillation_strict_threshold_and_quiet_priority():
    p = policy(deterioration=10, oscillation_change=.25)
    state = replace(initial_state(p, 100), recent_changes=(.5, -.5))
    _, d = transition(p, state, Checkpoint(125, 1, 1))
    assert not d.oscillating  # Exactly threshold is not significant.
    p = policy(deterioration=10, quiet_objective_change=1, quiet_checkpoints=1)
    state = replace(initial_state(p, 100), recent_changes=(.1, -.1))
    _, d = transition(p, state, Checkpoint(101, 0, 0))
    assert d.quiet and d.oscillating and d.reason == 'oscillation'


def test_deterioration_wins_even_when_quiet():
    p = policy(quiet_objective_change=1, quiet_checkpoints=1)
    _, d = transition(p, initial_state(p, 100), Checkpoint(102, 0, 0))
    assert d.quiet and d.reason == 'deterioration'


def test_safe_reference_and_effort_restoration():
    p = policy()
    state = replace(initial_state(p, 100), batch_size=16, safe_batch=128)
    state, d = transition(p, state, Checkpoint(102, 1, 1))
    assert state.batch_size == 128  # Restores more than merely doubling 16.
    state, d = transition(p, state, Checkpoint(90, 1, 1))
    assert state.safe_objective == 90 and state.safe_batch == 128
    # Relative to retained best, even if equal to the previous objective.
    state = replace(state, previous_objective=92)
    _, d = transition(p, state, Checkpoint(92, 1, 1))
    assert d.relative_change == 0 and d.deteriorated


def test_repeated_deterioration_at_max_is_not_a_change_or_reset():
    p = policy()
    state = replace(initial_state(p, 100), batch_size=256, recent_changes=(.02,))
    for i in range(5):
        state, d = transition(p, state, Checkpoint(102, 0, 0))
        assert d.action == 'increase' and d.reason == 'deterioration_at_bound'
        assert not d.changed and state.batch_size == 256
        assert state.quiet_count == i  # First .02 change is not quiet; later zeros are.
        assert state.checkpoints == i + 1
        assert len(state.recent_changes) == min(i + 2, 3)
    # No stopping counters exist here; a caller must reset them only on real changes.


def test_halving_rounding_clipping_and_decrease_cooldown():
    p = BatchPolicy.for_run(100, 17, minimum=9, maximum=20)
    rows = replay(p, [Checkpoint(100, 0, 0)] * 7)
    assert [s.batch_size for s, _ in rows] == [17, 17, 9, 9, 9, 9, 9]
    assert rows[5][1].reason == 'quiet_at_bound' and not rows[5][1].changed
    assert rows[6][0].quiet_count == 4  # Clipped no-op does not reset.
    state, d = transition(p, initial_state(p, 100), Checkpoint(102, 1, 1))
    assert state.batch_size == 20  # Doubling clips at max.


def test_zero_objective_and_floor():
    p = policy()
    rows = replay(p, [Checkpoint(0, 0, 0)] * 3, objective=0)
    assert rows[-1][0].batch_size == 32
    assert all(d.relative_change == 0 for _, d in rows)
    _, d = transition(p, initial_state(p, 0), Checkpoint(p.objective_floor*.02, 0, 0))
    assert d.deteriorated and d.relative_change == pytest.approx(.02)


def test_reproducibility_replay_and_rng_independence():
    p = policy()
    rng = np.random.default_rng(42)
    observations = tuple(Checkpoint(float(q), 0, 0) for q in rng.uniform(90, 110, 20))
    before = repr(rng.bit_generator.state)
    first, second = replay(p, observations), replay(p, observations)
    assert first == second and repr(rng.bit_generator.state) == before
    # Resume from a stored immutable state, with identical remaining decisions.
    state = first[7][0]
    for cp, expected in zip(observations[8:], first[8:]):
        state, decision = transition(p, state, cp)
        assert (state, decision) == expected
    with pytest.raises(FrozenInstanceError):
        p.minimum = 1
    with pytest.raises(FrozenInstanceError):
        state.batch_size = 1


def test_invalid_state_and_nonrepresentable_ratio():
    p = policy()
    with pytest.raises(ValueError):
        transition(p, replace(initial_state(p, 100), batch_size=1001), Checkpoint(100, 0, 0))
    with pytest.raises(ValueError):
        ControllerState(1, -1, 0, 1)
    with pytest.raises(ValueError):
        transition(policy(objective_floor=1e-300), initial_state(p, 0),
                   Checkpoint(1e300, 0, 0))
