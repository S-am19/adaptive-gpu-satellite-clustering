"""Fixed Mini-Batch K-Means using an Approximate Nearest Neighbor (ANN) backend.

Intermediate mini-batch assignments query the supplied or configured ANN
backend (HNSWCPUBackend by default). Centroids are updated via running
accumulated statistics, after which the ANN index is refreshed. Convergence
is evaluated on a fixed probe subset using exact assignments to prevent ANN
search noise from triggering false termination. Final labels and final SSE
are strictly computed using a 100% exact full-data pass.
"""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .assignments_ann import HNSWCPUBackend
from .assignments_exact import accumulate_cluster_stats, assign_exact
from .contracts import (
    AssignmentBackend,
    FitConfig,
    RunResult,
    validate_points_centroids,
)
from .minibatch import _PROBE_SIZE, _STABLE_CHECKS, _movement, _sample_batch_indices


def fit_minibatch_ann_cpu(X, C0, config, *, backend=None) -> RunResult:
    """Fit fixed Mini-Batch K-Means with an ANN assignment backend on CPU.

    config.batch_size must be supplied and <= N; config.ann_effort must be an
    integer >= 1. If backend is None, an HNSWCPUBackend is initialized with
    seed=config.seed and ann_effort=config.ann_effort.

    Stopping checks occur every probe_interval steps. Heuristic stability requires
    three consecutive checks with maximum centroid movement <= tolerance and
    unchanged exact probe labels. Final evaluation is strictly exact.
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
    if config.ann_effort is None:
        raise ValueError("Mini-Batch with ANN requires ann_effort to be an integer >= 1")
    if int(config.max_steps) * int(config.batch_size) > np.iinfo(np.int64).max:
        raise ValueError("Cumulative observation count exceeds int64 capacity")

    if backend is None:
        backend = HNSWCPUBackend(seed=config.seed, ann_effort=config.ann_effort)
    elif not all(callable(getattr(backend, name, None)) for name in ("prepare", "assign", "refresh")):
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

    # Initial index build
    maint_start = perf_counter()
    backend.prepare(centroids, effort=config.ann_effort)
    index_maintenance_time = perf_counter() - maint_start

    check_centroids = centroids.copy()
    monitor_start = perf_counter()
    previous_probe = assign_exact(probe, centroids)
    monitoring_time = perf_counter() - monitor_start

    history = []
    sampling_time = assignment_time = update_time = 0.0
    stable_checks = probe_checks = 0
    converged = False

    for step in range(1, int(config.max_steps) + 1):
        # 1. Sample mini-batch
        start = perf_counter()
        indices = _sample_batch_indices(rng, n, config.batch_size)
        batch = np.ascontiguousarray(X[indices])
        sampling_time += perf_counter() - start

        # 2. ANN batch assignment
        start = perf_counter()
        assignment = backend.assign(batch, centroids, effort=config.ann_effort)
        assignment_time += perf_counter() - start

        # 3. Running cluster statistics update
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
        update_time += perf_counter() - start

        # 4. Refresh ANN index with moved centroids
        start = perf_counter()
        backend.refresh(centroids)
        index_maintenance_time += perf_counter() - start

        entry = dict(
            step=step,
            batch_size=config.batch_size,
            points_processed=step * int(config.batch_size),
            centroid_displacement=movement,
            batch_sse_before_update=float(
                np.sum(assignment.squared_distances, dtype=np.float64)
            ),
            batch_objective_scope="sample only, at pre-update centroids (ANN)",
            cumulative_observations=int(counts.sum()),
            convergence_checked=False,
            ann_effort=int(config.ann_effort),
            index_update_time_s=assignment.index_update_time_s,
        )

        # 5. Checkpoint monitoring on fixed probe
        if step % config.probe_interval == 0:
            start = perf_counter()
            current_probe = assign_exact(probe, centroids)
            # Evaluate ANN on probe to track recall agreement against exact
            ann_probe = backend.assign(probe, centroids, effort=config.ann_effort)
            probe_recall = float(np.mean(ann_probe.labels == current_probe.labels))

            checkpoint_movement = _movement(centroids, check_centroids)
            changes = float(np.mean(current_probe.labels != previous_probe.labels))
            stable = checkpoint_movement <= config.tolerance and changes == 0.0
            stable_checks = stable_checks + 1 if stable else 0
            converged = stable_checks >= _STABLE_CHECKS
            probe_checks += 1

            entry.update(
                convergence_checked=True,
                checkpoint_displacement=checkpoint_movement,
                probe_sse=float(np.sum(current_probe.squared_distances, dtype=np.float64)),
                probe_ann_sse=float(np.sum(ann_probe.squared_distances, dtype=np.float64)),
                probe_recall=probe_recall,
                probe_objective_scope="fixed stopping probe only, at post-update centroids",
                probe_label_change_fraction=changes,
                stable_checks=stable_checks,
                convergence_satisfied=converged,
            )
            check_centroids = centroids.copy()
            previous_probe = current_probe
            monitoring_time += perf_counter() - start

        history.append(entry)
        if converged:
            break

    # 6. Final exact full-data assignment
    final_start = perf_counter()
    final = assign_exact(X, centroids)
    final_sse = float(np.sum(final.squared_distances, dtype=np.float64))
    final_time = perf_counter() - final_start

    # Sample ANN recall on probe at completion
    final_ann_probe = backend.assign(probe, centroids, effort=config.ann_effort)
    final_probe_recall = float(np.mean(final_ann_probe.labels == assign_exact(probe, centroids).labels))

    compute_time = perf_counter() - compute_start

    ann_timings = getattr(backend, "timings", {})
    metadata = {
        "algorithm": "fixed_minibatch_ann_cpu",
        "config": asdict(config),
        "final_exact_sse": final_sse,
        "final_probe_recall": final_probe_recall,
        "steps": step,
        "termination_reason": "tolerance" if converged else "max_steps",
        "convergence_satisfied": converged,
        "stopping_rule": "3 successive checkpoints: movement <= tolerance and unchanged probe labels",
        "stable_checks_required": _STABLE_CHECKS,
        "stable_checks": stable_checks,
        "probe_size": len(probe),
        "probe_checks": probe_checks,
        "probe_indices": probe_indices.tolist(),
        "probe_role": "overlapping training-population stopping diagnostic (exact evaluation)",
        "sampling_policy": "uniform without replacement within batch; independent draws across steps; full batch uses input order",
        "sampling_rng": "numpy.default_rng(seed)",
        "probe_rng": "numpy.default_rng(SeedSequence([seed,1]))",
        "cluster_observation_counts": counts.tolist(),
        "observation_counts_include_C0": False,
        "points_processed": step * int(config.batch_size),
        "assignment_points_processed": step * int(config.batch_size) + (2 * probe_checks + 3) * len(probe) + n,
        "ann_backend": backend.__class__.__name__,
        "ann_effort": int(config.ann_effort),
        "ann_timings": dict(ann_timings),
        "rebuild_count": ann_timings.get("rebuild_count", 0),
        "rebuild_s": ann_timings.get("rebuild_s", 0.0),
        "input_dtype": "float32",
        "centroid_dtype": "float32",
        "accumulation_dtype": "float64",
        "empty_cluster_policy": "retain",
        "tie_policy": "lowest_index for final exact evaluation; backend-specific during training",
        "initialization": "provided_C0",
        "history_objective_policy": "batch/probe objectives only; final full SSE exclusively in final_exact_sse",
        "timing_scope": (
            "compute includes sampling, batch assignments, accumulation/updates, index maintenance, monitoring, and final exact pass; "
            "component times are subsets, not additions to compute/wall"
        ),
    }

    timings = {
        "compute_s": compute_time,
        "sampling_s": sampling_time,
        "batch_assignment_s": assignment_time,
        "accumulation_update_s": update_time,
        "index_maintenance_s": index_maintenance_time,
        "monitoring_s": monitoring_time,
        "final_assignment_s": final_time,
        "wall_s": perf_counter() - started,
    }

    return RunResult(centroids, final.labels, history, timings, metadata)


# Convenience alias
fit_minibatch_ann = fit_minibatch_ann_cpu
