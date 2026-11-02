"""Pure policy for the experimental quality-guarded CUDA Mini-Batch method.

The policy sees only a fixed probe, its exact SSE trend, centroid movement,
and probe-label changes. It has no evaluation-target or reference-SSE input.
"""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class QualityGuardPolicy:
    minimum_batch: int
    maximum_batch: int
    initial_batch: int
    deterioration: float = 0.01
    stagnation_improvement: float = 0.001
    progress_improvement: float = 0.005
    movement_limit: float = 0.001
    label_change_limit: float = 0.005
    oscillation_change: float = 0.001
    required_checkpoints: int = 3
    dwell_checkpoints: int = 2
    adjustment_factor: int = 2
    objective_floor: float = 1e-12

    def __post_init__(self):
        for name in ("minimum_batch", "maximum_batch", "initial_batch",
                     "required_checkpoints", "dwell_checkpoints", "adjustment_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.adjustment_factor < 2 or not self.minimum_batch <= self.initial_batch <= self.maximum_batch:
            raise ValueError("invalid batch bounds or adjustment factor")
        for name in ("deterioration", "stagnation_improvement", "progress_improvement",
                     "movement_limit", "label_change_limit", "oscillation_change", "objective_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.objective_floor == 0 or self.label_change_limit > 1:
            raise ValueError("objective_floor must be positive and label_change_limit <= 1")
        if self.stagnation_improvement >= self.progress_improvement:
            raise ValueError("stagnation threshold must be below progress threshold")


@dataclass(frozen=True)
class QualityGuardState:
    batch_size: int
    previous_objective: float
    best_objective: float
    safe_batch: int
    stagnation_count: int = 0
    progress_count: int = 0
    dwell_remaining: int = 0
    recent_changes: tuple = ()
    checkpoints: int = 0


@dataclass(frozen=True)
class QualityGuardDecision:
    previous_batch: int
    next_batch: int
    action: str
    reason: str
    relative_change: float
    deteriorated: bool
    rollback: bool
    changed: bool


def initial_quality_guard_state(policy: QualityGuardPolicy, objective: float) -> QualityGuardState:
    _objective(objective)
    return QualityGuardState(policy.initial_batch, float(objective), float(objective),
                             policy.initial_batch)


def quality_guard_transition(policy: QualityGuardPolicy, state: QualityGuardState,
                             objective: float, movement: float,
                             label_change: float) -> tuple[QualityGuardState, QualityGuardDecision]:
    """Apply deterministic checkpoint policy; rollback means restore safe model state."""
    q = _objective(objective)
    movement = _nonnegative("movement", movement)
    label_change = _nonnegative("label_change", label_change)
    if label_change > 1 or not policy.minimum_batch <= state.batch_size <= policy.maximum_batch:
        raise ValueError("invalid label change or controller batch state")
    relative = (q - state.previous_objective) / max(state.previous_objective, policy.objective_floor)
    if not isfinite(relative):
        raise ValueError("probe objective change is not representable")
    changes = (state.recent_changes + (relative,))[-3:]
    oscillating = (len(changes) == 3 and
                   all(abs(x) > policy.oscillation_change for x in changes) and
                   all((a > 0) != (b > 0) for a, b in zip(changes, changes[1:])))
    deteriorated = q > state.best_objective + policy.deterioration * max(
        state.best_objective, policy.objective_floor)
    improvement = -relative
    stagnant = improvement <= policy.stagnation_improvement
    stagnation_count = state.stagnation_count + 1 if stagnant else 0
    progress = (improvement >= policy.progress_improvement and
                 movement <= policy.movement_limit and label_change <= policy.label_change_limit)
    progress_count = state.progress_count + 1 if progress else 0
    batch, action, reason, rollback = state.batch_size, "retain", "insufficient_evidence", False
    dwell = max(0, state.dwell_remaining - 1)
    if deteriorated:
        batch = min(policy.maximum_batch, max(batch * policy.adjustment_factor, state.safe_batch))
        action, reason, rollback = "increase", "probe_deterioration_restore_safe_state", True
    elif state.dwell_remaining:
        reason = "minimum_dwell"
    elif oscillating:
        batch = min(policy.maximum_batch, batch * policy.adjustment_factor)
        action, reason = "increase", "probe_objective_oscillation"
    elif stagnation_count >= policy.required_checkpoints:
        batch = min(policy.maximum_batch, batch * policy.adjustment_factor)
        action, reason = "increase", "probe_objective_stagnation"
    elif progress_count >= policy.required_checkpoints:
        batch = max(policy.minimum_batch, batch // policy.adjustment_factor)
        action, reason = "decrease", "sustained_progress_and_stability"
    changed = batch != state.batch_size
    if action != "retain" and not changed:
        action, reason = "retain", reason + "_at_bound"
    best = min(state.best_objective, q)
    safe_batch = state.batch_size if q < state.best_objective else state.safe_batch
    next_state = replace(state, batch_size=batch, previous_objective=q,
                         best_objective=best, safe_batch=safe_batch,
                         stagnation_count=0 if changed else stagnation_count,
                         progress_count=0 if changed else progress_count,
                         dwell_remaining=policy.dwell_checkpoints if changed else dwell,
                         recent_changes=() if changed else changes,
                         checkpoints=state.checkpoints + 1)
    return next_state, QualityGuardDecision(state.batch_size, batch, action, reason,
                                            relative, deteriorated, rollback, changed)


def _nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _objective(value):
    return _nonnegative("objective", value)
