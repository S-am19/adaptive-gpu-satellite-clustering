"""Pure checkpoint-level batch-size policy; no execution, RNG, or stopping logic."""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')


def _real(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be real')
    if not isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')


@dataclass(frozen=True)
class BatchPolicy:
    minimum: int
    maximum: int
    initial_batch: int
    deterioration: float = 0.01
    quiet_movement: float = 0.001
    quiet_label_change: float = 0.005
    quiet_objective_change: float = 0.001
    oscillation_change: float = 0.001
    quiet_checkpoints: int = 3
    objective_floor: float = 1e-12

    def __post_init__(self):
        for name in ('minimum', 'maximum', 'initial_batch', 'quiet_checkpoints'):
            _integer(name, getattr(self, name))
        if not self.minimum <= self.initial_batch <= self.maximum:
            raise ValueError('Require minimum <= initial_batch <= maximum')
        for name in ('deterioration', 'quiet_movement', 'quiet_label_change',
                     'quiet_objective_change', 'oscillation_change', 'objective_floor'):
            _real(name, getattr(self, name))
        if self.objective_floor == 0 or self.quiet_label_change > 1:
            raise ValueError('objective_floor must be positive; label threshold <= 1')

    @classmethod
    def for_run(cls, n, initial_batch, *, minimum=None, maximum=None, **thresholds):
        """Resolve bounds including the initial batch, even below 256/above 65536.

        Pass config.batch_size as initial_batch. Explicit bounds are validated,
        never silently clipped. No dependency on the shared FitConfig is needed.
        """
        _integer('n', n)
        _integer('initial_batch', initial_batch)
        if initial_batch > n:
            raise ValueError('initial_batch must not exceed N')
        minimum = min(int(initial_batch), 256, int(n)) if minimum is None else minimum
        maximum = max(int(initial_batch), min(int(n), 65536)) if maximum is None else maximum
        result = cls(minimum, maximum, initial_batch, **thresholds)
        if result.maximum > n:
            raise ValueError('maximum must not exceed N')
        return result


@dataclass(frozen=True)
class ControllerState:
    batch_size: int
    previous_objective: float
    safe_objective: float
    safe_batch: int
    quiet_count: int = 0
    recent_changes: tuple = ()
    checkpoints: int = 0

    def __post_init__(self):
        for name in ('batch_size', 'safe_batch'):
            _integer(name, getattr(self, name))
        for name in ('quiet_count', 'checkpoints'):
            _integer(name, getattr(self, name), 0)
        for name in ('previous_objective', 'safe_objective'):
            _real(name, getattr(self, name))
        if not isinstance(self.recent_changes, tuple) or len(self.recent_changes) > 3:
            raise ValueError('recent_changes must be a tuple of at most three values')
        for value in self.recent_changes:
            if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
                raise ValueError('recent changes must be finite real values')


@dataclass(frozen=True)
class Checkpoint:
    """Exact fixed-probe mean SSE, normalized movement, and label-change fraction."""
    objective: float
    movement: float
    label_change: float

    def __post_init__(self):
        for name in ('objective', 'movement', 'label_change'):
            _real(name, getattr(self, name))
        if self.label_change > 1:
            raise ValueError('label_change must be <= 1')


@dataclass(frozen=True)
class BatchDecision:
    previous_batch: int
    next_batch: int
    action: str
    reason: str
    relative_change: float
    deteriorated: bool
    oscillating: bool
    quiet: bool

    @property
    def changed(self):
        return self.previous_batch != self.next_batch


def initial_state(policy: BatchPolicy, objective: float) -> ControllerState:
    """Initialize from the exact probe objective at C0 (not a training update)."""
    return ControllerState(int(policy.initial_batch), objective, objective,
                           int(policy.initial_batch))


def transition(policy: BatchPolicy, state: ControllerState,
               checkpoint: Checkpoint) -> tuple[ControllerState, BatchDecision]:
    """Return new immutable state and replayable decision, without side effects.

    Priority: > deterioration threshold, three significant alternating changes,
    quiet streak, retain. Only actual size changes reset the change window and
    quiet streak. A clipped increase remains a requested increase with changed
    False; it never signals a real batch change or owns/reset stopping counters.
    No centroid/statistic rollback is requested. Future stopping confirmation
    must freeze adaptation externally rather than calling this function.
    """
    if not (policy.minimum <= state.batch_size <= policy.maximum and
            policy.minimum <= state.safe_batch <= policy.maximum):
        raise ValueError('State batch sizes must lie within policy bounds')
    q = float(checkpoint.objective)
    relative = (q - state.previous_objective) / max(state.previous_objective, policy.objective_floor)
    if not isfinite(relative):
        raise ValueError('Objective ratio is not representable; rescale objectives')
    changes = (state.recent_changes + (relative,))[-3:]
    oscillating = (len(changes) == 3 and
                   all(abs(v) > policy.oscillation_change for v in changes) and
                   all((a > 0) != (b > 0) for a, b in zip(changes, changes[1:])))
    deteriorated = q > state.safe_objective + policy.deterioration * max(
        state.safe_objective, policy.objective_floor)
    quiet = (checkpoint.movement <= policy.quiet_movement and
             checkpoint.label_change <= policy.quiet_label_change and
             abs(relative) <= policy.quiet_objective_change)
    quiet_count = state.quiet_count + 1 if quiet else 0
    batch = int(state.batch_size)
    next_batch, action, reason = batch, 'retain', 'insufficient_quiet_checkpoints' if quiet else 'active'
    if deteriorated:
        next_batch = min(int(policy.maximum), max(2 * batch, int(state.safe_batch)))
        action, reason = 'increase', 'deterioration'
    elif oscillating:
        next_batch = min(int(policy.maximum), 2 * batch)
        action, reason = 'increase', 'oscillation'
    elif quiet_count >= policy.quiet_checkpoints:
        next_batch = max(int(policy.minimum), batch // 2)
        action, reason = 'decrease', 'quiet'
    if next_batch == batch and action != 'retain':
        reason += '_at_bound'
    safe_objective, safe_batch = state.safe_objective, state.safe_batch
    if q < safe_objective:
        safe_objective, safe_batch = q, batch  # This batch produced the checkpoint.
    new_state = replace(state, batch_size=next_batch, previous_objective=q,
                        safe_objective=safe_objective, safe_batch=safe_batch,
                        quiet_count=0 if next_batch != batch else quiet_count,
                        recent_changes=() if next_batch != batch else changes,
                        checkpoints=state.checkpoints + 1)
    return new_state, BatchDecision(batch, next_batch, action, reason, relative,
                                    deteriorated, oscillating, quiet)
