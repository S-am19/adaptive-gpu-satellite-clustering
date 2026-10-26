"""Pure quality-trend batch policy for experimental Adaptive Mini-Batch V2.

This controller only consumes fixed-probe measurements. It deliberately has no
quality target or exact-reference input; those belong to evaluation code.
"""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _real(name, value, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _signed_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class AdaptiveV2Policy:
    minimum: int
    maximum: int
    initial_batch: int
    deterioration: float = 0.01
    stagnation_improvement: float = 0.001
    progress_improvement: float = 0.005
    quiet_movement: float = 0.001
    quiet_label_change: float = 0.005
    required_checkpoints: int = 3
    objective_floor: float = 1e-12
    adjustment_factor: int = 2

    def __post_init__(self):
        for name in ("minimum", "maximum", "initial_batch", "required_checkpoints"):
            _integer(name, getattr(self, name), 1)
        _integer("adjustment_factor", self.adjustment_factor, 2)
        if not self.minimum <= self.initial_batch <= self.maximum:
            raise ValueError("Require minimum <= initial_batch <= maximum")
        for name in ("deterioration", "stagnation_improvement", "progress_improvement",
                     "quiet_movement", "quiet_label_change", "objective_floor"):
            _real(name, getattr(self, name))
        if self.objective_floor == 0:
            raise ValueError("objective_floor must be positive")
        if self.quiet_label_change > 1:
            raise ValueError("quiet_label_change must be <= 1")
        if self.stagnation_improvement >= self.progress_improvement:
            raise ValueError("stagnation threshold must be below progress threshold")

    @classmethod
    def for_run(cls, n, initial_batch, *, minimum=None, maximum=None, **thresholds):
        _integer("n", n, 1)
        _integer("initial_batch", initial_batch, 1)
        if initial_batch > n:
            raise ValueError("initial_batch must not exceed N")
        low = min(initial_batch, 256, n) if minimum is None else minimum
        high = max(initial_batch, min(n, 65536)) if maximum is None else maximum
        result = cls(low, high, initial_batch, **thresholds)
        if result.maximum > n:
            raise ValueError("maximum must not exceed N")
        return result


@dataclass(frozen=True)
class AdaptiveV2State:
    batch_size: int
    previous_probe_sse: float
    best_probe_sse: float
    safe_batch_size: int
    stagnation_count: int = 0
    progress_count: int = 0
    recent_changes: tuple[float, ...] = ()
    checkpoints: int = 0

    def __post_init__(self):
        for name in ("batch_size", "safe_batch_size"):
            _integer(name, getattr(self, name), 1)
        for name in ("stagnation_count", "progress_count", "checkpoints"):
            _integer(name, getattr(self, name), 0)
        for name in ("previous_probe_sse", "best_probe_sse"):
            _real(name, getattr(self, name))
        if not isinstance(self.recent_changes, tuple) or len(self.recent_changes) > 3:
            raise ValueError("recent_changes must be a tuple of at most three values")
        for change in self.recent_changes:
            _signed_real("recent change", change)


@dataclass(frozen=True)
class AdaptiveV2Checkpoint:
    """Fixed-probe mean squared distance and normalized movement/change signals."""

    probe_sse: float
    normalized_movement: float
    label_change_fraction: float

    def __post_init__(self):
        for name in ("probe_sse", "normalized_movement", "label_change_fraction"):
            _real(name, getattr(self, name))
        if self.label_change_fraction > 1:
            raise ValueError("label_change_fraction must be <= 1")


@dataclass(frozen=True)
class AdaptiveV2Decision:
    previous_batch: int
    next_batch: int
    action: str
    reason: str
    relative_probe_change: float
    deteriorated: bool
    oscillating: bool
    stagnation_count: int
    progress_count: int

    @property
    def changed(self):
        return self.previous_batch != self.next_batch


def initial_v2_state(policy: AdaptiveV2Policy, initial_probe_sse: float) -> AdaptiveV2State:
    if not isinstance(policy, AdaptiveV2Policy):
        raise TypeError("policy must be AdaptiveV2Policy")
    _real("initial_probe_sse", initial_probe_sse)
    return AdaptiveV2State(policy.initial_batch, float(initial_probe_sse),
                           float(initial_probe_sse), policy.initial_batch)


def transition_v2(policy: AdaptiveV2Policy, state: AdaptiveV2State,
                  checkpoint: AdaptiveV2Checkpoint):
    """Pure, deterministic transition; no target/reference value is accepted.

    Priority: >1% deterioration, significant three-checkpoint oscillation,
    three stagnant checkpoints, three sustained improving/stable checkpoints,
    retain. A shrink requires both objective progress and stable movement/labels;
    small movement alone can never request one. Stagnation grows the batch even
    if movement is small, avoiding repeated shrinkage on a poor plateau.

    Counters reset only after an actual batch-size change. Requests clipped at a
    bound are logged as no-ops and do not erase evidence or fabricate a change.
    """
    if not isinstance(policy, AdaptiveV2Policy) or not isinstance(state, AdaptiveV2State):
        raise TypeError("policy and state must use the Adaptive V2 dataclasses")
    if not isinstance(checkpoint, AdaptiveV2Checkpoint):
        raise TypeError("checkpoint must be AdaptiveV2Checkpoint")
    if not policy.minimum <= state.batch_size <= policy.maximum:
        raise ValueError("state batch size is outside policy bounds")

    q = float(checkpoint.probe_sse)
    denominator = max(state.previous_probe_sse, policy.objective_floor)
    relative = (q - state.previous_probe_sse) / denominator
    if not isfinite(relative):
        raise ValueError("probe objective change is not representable")
    changes = (state.recent_changes + (relative,))[-3:]
    oscillating = (len(changes) == 3 and
                   all(abs(v) > policy.stagnation_improvement for v in changes) and
                   all((a > 0) != (b > 0) for a, b in zip(changes, changes[1:])))
    deteriorated = q > state.best_probe_sse + policy.deterioration * max(
        state.best_probe_sse, policy.objective_floor)
    improvement = -relative
    stagnant = improvement <= policy.stagnation_improvement
    stagnation_count = min(policy.required_checkpoints, state.stagnation_count + 1) if stagnant else 0
    stable_progress = (improvement >= policy.progress_improvement and
                       checkpoint.normalized_movement <= policy.quiet_movement and
                       checkpoint.label_change_fraction <= policy.quiet_label_change)
    progress_count = min(policy.required_checkpoints, state.progress_count + 1) if stable_progress else 0

    current = int(state.batch_size)
    requested = current
    action, reason = "retain", "active_progress_or_insufficient_evidence"
    if deteriorated:
        requested = min(policy.maximum, max(current * policy.adjustment_factor,
                                            state.safe_batch_size))
        action, reason = "increase", "probe_deterioration_restore_safe_batch"
    elif oscillating:
        requested = min(policy.maximum, current * policy.adjustment_factor)
        action, reason = "increase", "probe_objective_oscillation"
    elif stagnation_count >= policy.required_checkpoints:
        requested = min(policy.maximum, current * policy.adjustment_factor)
        action, reason = "increase", "probe_objective_stagnation"
    elif progress_count >= policy.required_checkpoints:
        requested = max(policy.minimum, current // policy.adjustment_factor)
        action, reason = "decrease", "sustained_objective_progress_with_stable_assignments"

    clipped = requested == current and action != "retain"
    if clipped:
        reason += "_at_bound"
        action = "retain"
    changed = requested != current
    best = min(state.best_probe_sse, q)
    safe_batch = current if q < state.best_probe_sse else state.safe_batch_size
    next_state = replace(
        state,
        batch_size=requested,
        previous_probe_sse=q,
        best_probe_sse=best,
        safe_batch_size=safe_batch,
        stagnation_count=0 if changed else stagnation_count,
        progress_count=0 if changed else progress_count,
        recent_changes=() if changed else changes,
        checkpoints=state.checkpoints + 1,
    )
    decision = AdaptiveV2Decision(current, requested, action, reason, relative,
                                  deteriorated, oscillating, stagnation_count,
                                  progress_count)
    return next_state, decision
