"""Combined adaptive Mini-Batch K-Means integrating batch sizing and ANN effort.

Supports three distinct ablation modes:
- 'batch_only' (G3): Adapts mini-batch size while keeping search effort fixed.
- 'ann_only' (G4): Adapts ANN search effort (efSearch) with a fixed batch size.
- 'combined' (G5): Jointly adapts both batch size and ANN effort simultaneously.

A frozen confirmation phase at maximum batch size / high search effort guarantees
that heuristic probe stability is robust before stopping. Final clustering
labels and SSE are strictly computed via a 100% exact full-data pass.
"""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .adaptive_ann_controller import (
    ANNEffortPolicy,
    ANNEffortState,
    initial_ann_state,
    transition_ann,
)
from .adaptive_controller import (
    BatchPolicy,
    Checkpoint,
    ControllerState,
    initial_state,
    transition,
)
from .assignments_ann import HNSWCPUBackend
from .assignments_exact import ExactCPUBackend, accumulate_cluster_stats, assign_exact
from .contracts import (
    AssignmentBackend,
    FitConfig,
    RunResult,
    validate_points_centroids,
)
from .minibatch import _PROBE_SIZE, _STABLE_CHECKS, _movement, _sample_batch_indices


def fit_adaptive_minibatch(
    X,
    C0,
    config,
    *,
    backend=None,
    batch_policy=None,
    ann_policy=None,
    mode="combined",
) -> RunResult:
    """Run adaptive Mini-Batch K-Means supporting G3, G4, and G5 ablations.

    mode must be one of:
      - 'batch_only': adapts batch size using batch_policy (exact or fixed ANN)
      - 'ann_only': adapts ann_effort using ann_policy (fixed batch size)
      - 'combined': jointly adapts both batch size and ANN search effort

    Confirmation requires three successive qualifying checkpoints with movement <= tolerance,
    zero probe label changes, and no objective deterioration relative to confirmation entry.
    """
    started = perf_counter()
    if not isinstance(config, FitConfig):
        raise TypeError("config must be FitConfig")
    validate_points_centroids(X, C0)
    n, d = X.shape
    k = config.k
    if len(C0) != k or k > n:
        raise ValueError("C0 rows must equal config.k, with k <= N")
    if config.batch_size is None or config.batch_size > n:
        raise ValueError("batch_size must be supplied and <= N")
    if mode not in ("batch_only", "ann_only", "combined"):
        raise ValueError("mode must be 'batch_only', 'ann_only', or 'combined'")

    # Configure policies
    use_adaptive_batch = mode in ("batch_only", "combined")
    use_adaptive_ann = mode in ("ann_only", "combined")

    if use_adaptive_batch:
        if batch_policy is None:
            batch_policy = BatchPolicy.for_run(n, config.batch_size)
        elif not isinstance(batch_policy, BatchPolicy):
            raise TypeError("batch_policy must be BatchPolicy or None")
        if batch_policy.initial_batch != config.batch_size or batch_policy.maximum > n:
            raise ValueError("batch_policy must match initial batch and maximum <= N")
    else:
        batch_policy = None

    if use_adaptive_ann:
        initial_effort = config.ann_effort if config.ann_effort is not None else 64
        if ann_policy is None:
            ann_policy = ANNEffortPolicy(
                levels=(64, 32, 16, 8),
                initial_effort=initial_effort if initial_effort in (64, 32, 16, 8) else 64,
            )
        elif not isinstance(ann_policy, ANNEffortPolicy):
            raise TypeError("ann_policy must be ANNEffortPolicy or None")
    else:
        ann_policy = None

    # Resolve backend
    if backend is None:
        if mode == "batch_only" and config.ann_effort is None:
            backend = ExactCPUBackend()
        else:
            effort = ann_policy.initial_effort if ann_policy else (config.ann_effort or 16)
            backend = HNSWCPUBackend(seed=config.seed, ann_effort=effort)
    elif not hasattr(backend, "assign"):
        raise TypeError("backend must implement the AssignmentBackend protocol")

    compute_start = perf_counter()
    rng = np.random.default_rng(config.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    probe_size = min(n, _PROBE_SIZE)
    probe_indices = np.sort(probe_rng.choice(n, size=probe_size, replace=False))
    probe = np.ascontiguousarray(X[probe_indices])

    centroids = C0.copy()
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)

    # Initial backend preparation
    initial_effort = ann_policy.initial_effort if ann_policy else config.ann_effort
    maint_start = perf_counter()
    backend.prepare(centroids, effort=initial_effort)
    index_maintenance_time = perf_counter() - maint_start

    # Initial probe objective for controller initialization
    start = perf_counter()
    initial_probe = assign_exact(probe, centroids)
    initial_objective = float(np.mean(initial_probe.squared_distances, dtype=np.float64))
    monitoring_time = perf_counter() - start

    # Initialize controller states
    probe64 = probe.astype(np.float64)
    spread_sq = float(np.mean(np.sum((probe64 - probe64.mean(axis=0)) ** 2, axis=1)))
    scale = max(float(np.sqrt(spread_sq)), 1e-12)
    floor = 1e-12 * scale * scale

    if use_adaptive_batch:
        batch_policy = replace(batch_policy, objective_floor=floor)
        b_state = initial_state(batch_policy, initial_objective)
    else:
        b_state = None

    if use_adaptive_ann:
        ann_policy = replace(ann_policy, objective_floor=floor)
        a_state = initial_ann_state(ann_policy, initial_objective)
    else:
        a_state = None

    check_centroids = centroids.copy()
    previous_probe = initial_probe
    history = []
    sampling_time = assignment_time = update_time = controller_time = 0.0
    stable_checks = confirmation_checks = probe_checks = 0
    observations = 0
    confirming = converged = False
    confirmation_objective = None

    for step in range(1, int(config.max_steps) + 1):
        # Current step parameters
        batch_size = int(b_state.batch_size) if b_state else int(config.batch_size)
        effort = int(a_state.effort) if a_state else config.ann_effort
        phase = "confirmation" if confirming else "adapting"

        # 1. Sample batch
        start = perf_counter()
        indices = _sample_batch_indices(rng, n, batch_size)
        batch = np.ascontiguousarray(X[indices])
        sampling_time += perf_counter() - start

        # 2. Assign batch
        start = perf_counter()
        assignment = backend.assign(batch, centroids, effort=effort)
        assignment_time += perf_counter() - start

        # 3. Accumulate stats and update centroids
        start = perf_counter()
        stats = accumulate_cluster_stats(batch, assignment.labels, k)
        previous_centroids = centroids.copy()
        sums += stats.sums
        counts += stats.counts
        active = stats.counts > 0
        centroids[active] = (sums[active] / counts[active, None]).astype(np.float32)
        if not np.isfinite(sums).all() or not np.isfinite(centroids).all():
            raise FloatingPointError("Nonfinite accumulated centroid statistics")
        movement = _movement(centroids, previous_centroids)
        observations += batch_size
        update_time += perf_counter() - start

        # 4. Refresh backend
        start = perf_counter()
        backend.refresh(centroids)
        index_maintenance_time += perf_counter() - start

        entry = dict(
            step=step,
            phase=phase,
            mode=mode,
            batch_size=batch_size,
            ann_effort=effort,
            points_processed=observations,
            cumulative_observations=int(counts.sum()),
            centroid_displacement=movement,
            batch_sse_before_update=float(
                np.sum(assignment.squared_distances, dtype=np.float64)
            ),
            batch_objective_scope="sample only, at pre-update centroids",
            convergence_checked=False,
            index_update_time_s=assignment.index_update_time_s,
        )

        # 5. Checkpoint monitoring
        if step % config.probe_interval == 0:
            start = perf_counter()
            current_probe = assign_exact(probe, centroids)
            probe_mean_sse = float(np.mean(current_probe.squared_distances, dtype=np.float64))
            checkpoint_movement = _movement(centroids, check_centroids)
            norm_movement = checkpoint_movement / scale
            label_changes = float(np.mean(current_probe.labels != previous_probe.labels))
            monitoring_time += perf_counter() - start

            # Evaluate ANN on probe to track empirical recall
            if hasattr(backend, "assign") and not isinstance(backend, ExactCPUBackend):
                ann_probe = backend.assign(probe, centroids, effort=effort)
                probe_recall = float(np.mean(ann_probe.labels == current_probe.labels))
            else:
                probe_recall = 1.0

            probe_checks += 1
            checkpoint = Checkpoint(probe_mean_sse, norm_movement, label_changes)

            if confirming:
                # In confirmation phase: adaptation is frozen at maximum configuration
                max_allowed = confirmation_objective * (1.0 + (batch_policy.deterioration if batch_policy else 0.01))
                qualifies = (
                    checkpoint_movement <= config.tolerance
                    and label_changes == 0.0
                    and probe_mean_sse <= max_allowed
                )
                confirmation_checks = confirmation_checks + 1 if qualifies else 0
                converged = confirmation_checks >= _STABLE_CHECKS
                entry.update(
                    convergence_checked=True,
                    checkpoint_displacement=checkpoint_movement,
                    probe_mean_sse=probe_mean_sse,
                    probe_recall=probe_recall,
                    probe_label_change_fraction=label_changes,
                    confirmation_checks=confirmation_checks,
                    convergence_satisfied=converged,
                )
            else:
                # In adapting phase: transition state machines
                start = perf_counter()
                b_decision = a_decision = None

                if use_adaptive_batch:
                    b_state, b_decision = transition(batch_policy, b_state, checkpoint)
                if use_adaptive_ann:
                    a_state, a_decision = transition_ann(ann_policy, a_state, checkpoint)

                controller_time += perf_counter() - start

                # Check if stability qualifies for confirmation phase
                stable = checkpoint_movement <= config.tolerance and label_changes == 0.0
                stable_checks = stable_checks + 1 if stable else 0

                # If 3 stable adapting checkpoints achieved: enter confirmation phase
                if stable_checks >= _STABLE_CHECKS:
                    confirming = True
                    confirmation_objective = probe_mean_sse
                    confirmation_checks = 0
                    if use_adaptive_batch:
                        b_state = replace(b_state, batch_size=int(batch_policy.maximum))
                    if use_adaptive_ann:
                        a_state = replace(a_state, effort=int(ann_policy.maximum))

                entry.update(
                    convergence_checked=True,
                    checkpoint_displacement=checkpoint_movement,
                    probe_mean_sse=probe_mean_sse,
                    probe_recall=probe_recall,
                    probe_label_change_fraction=label_changes,
                    stable_checks=stable_checks,
                    convergence_satisfied=False,
                    batch_decision=asdict(b_decision) if b_decision else None,
                    ann_decision=asdict(a_decision) if a_decision else None,
                )

            check_centroids = centroids.copy()
            previous_probe = current_probe

        history.append(entry)
        if converged:
            break

    # 6. Final exact full-data evaluation
    final_start = perf_counter()
    final = assign_exact(X, centroids)
    final_sse = float(np.sum(final.squared_distances, dtype=np.float64))
    final_time = perf_counter() - final_start
    compute_time = perf_counter() - compute_start

    ann_timings = getattr(backend, "timings", {})
    metadata = {
        "algorithm": f"adaptive_minibatch_{mode}",
        "mode": mode,
        "config": asdict(config),
        "final_exact_sse": final_sse,
        "steps": step,
        "termination_reason": "tolerance" if converged else "max_steps",
        "convergence_satisfied": converged,
        "confirming_phase_entered": confirming,
        "confirmation_checks": confirmation_checks,
        "stable_checks": stable_checks,
        "probe_size": len(probe),
        "probe_checks": probe_checks,
        "probe_indices": probe_indices.tolist(),
        "points_processed": observations,
        "cumulative_observations": int(counts.sum()),
        "cluster_observation_counts": counts.tolist(),
        "ann_backend": backend.__class__.__name__,
        "ann_timings": dict(ann_timings),
        "rebuild_count": ann_timings.get("rebuild_count", 0),
        "rebuild_s": ann_timings.get("rebuild_s", 0.0),
        "input_dtype": "float32",
        "centroid_dtype": "float32",
        "accumulation_dtype": "float64",
        "empty_cluster_policy": "retain",
        "tie_policy": "lowest_index",
        "initialization": "provided_C0",
        "history_objective_policy": "batch/probe objectives only; final full SSE exclusively in final_exact_sse",
        "timing_scope": (
            "compute includes sampling, batch assignments, accumulation/updates, controller logic, "
            "monitoring and final exact pass; component times are subsets, not additions to compute/wall"
        ),
    }

    timings = {
        "compute_s": compute_time,
        "sampling_s": sampling_time,
        "batch_assignment_s": assignment_time,
        "accumulation_update_s": update_time,
        "index_maintenance_s": index_maintenance_time,
        "controller_s": controller_time,
        "monitoring_s": monitoring_time,
        "final_assignment_s": final_time,
        "wall_s": perf_counter() - started,
    }

    return RunResult(centroids, final.labels, history, timings, metadata)
