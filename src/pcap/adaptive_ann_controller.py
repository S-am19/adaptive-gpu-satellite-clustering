"""Pure checkpoint-level ANN search-effort policy; no execution, RNG, or stopping logic.

Dynamically scales ANN search depth (e.g. HNSW efSearch) based on convergence
diagnostics. When normalized centroid movement and probe label changes remain
quiet across multiple consecutive checkpoints, search effort is stepped down.
If probe SSE deteriorates by more than the allowed tolerance relative to the
retained safe checkpoint, effort is rolled back to the last safe level.
"""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _real(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class ANNEffortPolicy:
    """Configuration for ANN search effort adaptation across checkpoints.

    levels must be a non-empty tuple of distinct positive integers in descending
    order (e.g. (64, 32, 16, 8)). initial_effort must belong to levels.
    """
    levels: tuple[int, ...] = (64, 32, 16, 8)
    initial_effort: int = 64
    deterioration: float = 0.01
    quiet_movement: float = 0.001
    quiet_label_change: float = 0.005
    quiet_objective_change: float = 0.001
    quiet_checkpoints: int = 3
    objective_floor: float = 1e-12

    def __post_init__(self):
        if not isinstance(self.levels, (tuple, list)) or len(self.levels) == 0:
            raise ValueError("levels must be a non-empty sequence of effort values")
        levels_tuple = tuple(int(v) for v in self.levels)
        for val in levels_tuple:
            _integer("effort level", val, minimum=1)
        if len(set(levels_tuple)) != len(levels_tuple):
            raise ValueError("levels must contain distinct values")
        if levels_tuple != tuple(sorted(levels_tuple, reverse=True)):
            raise ValueError("levels must be sorted in strictly descending order")
        object.__setattr__(self, "levels", levels_tuple)

        _integer("initial_effort", self.initial_effort, minimum=1)
        if self.initial_effort not in self.levels:
            raise ValueError(f"initial_effort {self.initial_effort} must be in levels {self.levels}")
        _integer("quiet_checkpoints", self.quiet_checkpoints, minimum=1)

        for name in (
            "deterioration",
            "quiet_movement",
            "quiet_label_change",
            "quiet_objective_change",
            "objective_floor",
        ):
            _real(name, getattr(self, name))
        if self.objective_floor == 0 or self.quiet_label_change > 1:
            raise ValueError("objective_floor must be positive; label threshold <= 1")

    @property
    def maximum(self) -> int:
        return self.levels[0]

    @property
    def minimum(self) -> int:
        return self.levels[-1]


@dataclass(frozen=True)
class ANNEffortState:
    effort: int
    previous_objective: float
    safe_objective: float
    safe_effort: int
    quiet_count: int = 0
    checkpoints: int = 0

    def __post_init__(self):
        for name in ("effort", "safe_effort"):
            _integer(name, getattr(self, name))
        for name in ("quiet_count", "checkpoints"):
            _integer(name, getattr(self, name), 0)
        for name in ("previous_objective", "safe_objective"):
            _real(name, getattr(self, name))


@dataclass(frozen=True)
class ANNDecision:
    previous_effort: int
    next_effort: int
    action: str
    reason: str
    relative_change: float
    deteriorated: bool
    quiet: bool

    @property
    def changed(self) -> bool:
        return self.previous_effort != self.next_effort


def initial_ann_state(policy: ANNEffortPolicy, objective: float) -> ANNEffortState:
    """Initialize from the exact probe objective at C0."""
    return ANNEffortState(
        effort=int(policy.initial_effort),
        previous_objective=float(objective),
        safe_objective=float(objective),
        safe_effort=int(policy.initial_effort),
    )


def transition_ann(
    policy: ANNEffortPolicy, state: ANNEffortState, checkpoint
) -> tuple[ANNEffortState, ANNDecision]:
    """Pure transition function evaluating ANN search effort adaptation.

    Priority:
    1. Deterioration: probe objective exceeds safe_objective * (1 + deterioration)
       -> roll back to safe_effort.
    2. Quiet streak: normalized movement, label change, and relative objective
       change remain quiet for quiet_checkpoints consecutive checks
       -> step down to the next lower effort level.
    3. Retain: otherwise maintain current effort level.
    """
    if state.effort not in policy.levels or state.safe_effort not in policy.levels:
        raise ValueError("State efforts must belong to policy levels")

    q = float(checkpoint.objective)
    denom = max(state.previous_objective, policy.objective_floor)
    relative = (q - state.previous_objective) / denom
    if not isfinite(relative):
        raise ValueError("Objective ratio is not representable")

    deteriorated = q > state.safe_objective + policy.deterioration * max(
        state.safe_objective, policy.objective_floor
    )

    quiet = (
        checkpoint.movement <= policy.quiet_movement
        and checkpoint.label_change <= policy.quiet_label_change
        and abs(relative) <= policy.quiet_objective_change
    )
    quiet_count = state.quiet_count + 1 if quiet else 0

    effort = state.effort
    current_idx = policy.levels.index(effort)

    if deteriorated:
        if effort < state.safe_effort:
            next_effort = state.safe_effort
            action, reason = "rollback", "deterioration"
        elif current_idx > 0:
            next_effort = policy.levels[current_idx - 1]
            action, reason = "increase", "deterioration"
        else:
            next_effort = effort
            action, reason = "retain", "deterioration_at_maximum"
    elif quiet_count >= policy.quiet_checkpoints:
        if current_idx < len(policy.levels) - 1:
            next_effort = policy.levels[current_idx + 1]
            action, reason = "decrease", "quiet"
        else:
            next_effort = effort
            action, reason = "retain", "quiet_at_minimum"
    else:
        next_effort = effort
        action, reason = "retain", "insufficient_quiet_checkpoints" if quiet else "active"

    safe_objective, safe_effort = state.safe_objective, state.safe_effort
    if q < safe_objective:
        safe_objective = q
        safe_effort = effort

    new_state = replace(
        state,
        effort=next_effort,
        previous_objective=q,
        safe_objective=safe_objective,
        safe_effort=safe_effort,
        quiet_count=0 if next_effort != effort else quiet_count,
        checkpoints=state.checkpoints + 1,
    )

    decision = ANNDecision(
        previous_effort=effort,
        next_effort=next_effort,
        action=action,
        reason=reason,
        relative_change=relative,
        deteriorated=deteriorated,
        quiet=quiet,
    )

    return new_state, decision
