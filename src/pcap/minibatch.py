"""Fixed CPU Mini-Batch K-Means with exact batch assignment, no CUDA/ANN.

Each batch is drawn uniformly without replacement; separate updates may revisit
points. For batch_size=N the batch is all rows in input order. Observation sums
and counts persist across batches and include repeats, not the supplied C0.
"""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .assignments_exact import ExactCPUBackend, accumulate_cluster_stats, assign_exact
from .contracts import FitConfig, RunResult, validate_points_centroids

_STABLE_CHECKS = 3
_PROBE_SIZE = 1024


def _sample_batch_indices(rng, n, batch_size):
    if batch_size == n:
        return np.arange(n)
    return rng.choice(n, size=batch_size, replace=False)


def _movement(current, previous):
    delta = current.astype(np.float64) - previous.astype(np.float64)
    return float(np.sqrt(np.max(np.sum(delta * delta, axis=1))))


def fit_minibatch_cpu(X, C0, config) -> RunResult:
    """Fit fixed batches using the existing CUDA-independent configuration.

    batch_size is required and must not exceed N; ann_effort must be None.
    Stopping checks occur every probe_interval updates. Three successive checks
    must have max centroid movement since the previous check <= tolerance AND
    unchanged exact labels on a fixed seeded probe (up to 1024 input rows).
    The probe overlaps the training population: it is a stopping diagnostic,
    not an independent validation set or a full-dataset quality guarantee.
    Batch/probe objectives can increase and never substitute for final SSE.
    Cold CPU assignment/accumulation JIT is included in measured durations.
    """
    started = perf_counter()
    if not isinstance(config, FitConfig):
        raise TypeError('config must be FitConfig')
    validate_points_centroids(X, C0)
    n, d = X.shape
    if len(C0) != config.k or config.k > n:
        raise ValueError('C0 rows must equal config.k, with k <= N')
    if config.batch_size is None or config.batch_size > n:
        raise ValueError('batch_size must be supplied and <= N')
    if config.ann_effort is not None:
        raise ValueError('Fixed exact CPU Mini-Batch requires ann_effort=None')
    # FitConfig accepts NumPy Integral values; their multiplication can wrap
    # before comparison. Python integers keep this capacity check exact.
    if int(config.max_steps) * int(config.batch_size) > np.iinfo(np.int64).max:
        raise ValueError('Cumulative observation count exceeds int64 capacity')
    compute_start = perf_counter()
    rng = np.random.default_rng(config.seed)
    # A separate random stream prevents probe selection changing batch draws.
    probe_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    probe_indices = np.sort(probe_rng.choice(n, size=min(n, _PROBE_SIZE), replace=False))
    probe = np.ascontiguousarray(X[probe_indices])
    centroids = C0.copy()
    sums = np.zeros((config.k, d), np.float64)
    counts = np.zeros(config.k, np.int64)
    backend = ExactCPUBackend()
    backend.prepare(centroids)
    check_centroids = centroids.copy()
    monitor_start = perf_counter()
    previous_probe = backend.assign(probe, centroids)
    monitoring_time = perf_counter() - monitor_start
    history = []
    sampling_time = assignment_time = update_time = 0.0
    stable_checks = probe_checks = 0
    converged = False
    for step in range(1, int(config.max_steps) + 1):
        start = perf_counter()
        indices = _sample_batch_indices(rng, n, config.batch_size)
        batch = np.ascontiguousarray(X[indices])
        sampling_time += perf_counter() - start
        start = perf_counter()
        assignment = backend.assign(batch, centroids)
        assignment_time += perf_counter() - start
        start = perf_counter()
        stats = accumulate_cluster_stats(batch, assignment.labels, config.k)
        previous = centroids.copy()
        sums += stats.sums
        counts += stats.counts
        active = stats.counts > 0
        centroids[active] = (sums[active] / counts[active, None]).astype(np.float32)
        if not np.isfinite(sums).all() or not np.isfinite(centroids).all():
            raise FloatingPointError('Nonfinite accumulated centroid statistics')
        movement = _movement(centroids, previous)
        backend.refresh(centroids)
        update_time += perf_counter() - start
        entry = dict(step=step, batch_size=config.batch_size,
                     points_processed=step * int(config.batch_size),
                     centroid_displacement=movement,
                     batch_sse_before_update=float(np.sum(assignment.squared_distances, dtype=np.float64)),
                     batch_objective_scope='sample only, at pre-update centroids',
                     cumulative_observations=int(counts.sum()), convergence_checked=False)
        if step % config.probe_interval == 0:
            start = perf_counter()
            current_probe = backend.assign(probe, centroids)
            checkpoint_movement = _movement(centroids, check_centroids)
            changes = float(np.mean(current_probe.labels != previous_probe.labels))
            stable = checkpoint_movement <= config.tolerance and changes == 0.0
            stable_checks = stable_checks + 1 if stable else 0
            converged = stable_checks >= _STABLE_CHECKS
            probe_checks += 1
            entry.update(convergence_checked=True, checkpoint_displacement=checkpoint_movement,
                         probe_sse=float(np.sum(current_probe.squared_distances, dtype=np.float64)),
                         probe_objective_scope='fixed stopping probe only, at post-update centroids',
                         probe_label_change_fraction=changes, stable_checks=stable_checks,
                         convergence_satisfied=converged)
            check_centroids = centroids.copy()
            previous_probe = current_probe
            monitoring_time += perf_counter() - start
        history.append(entry)
        if converged:
            break
    final_start = perf_counter()
    final = assign_exact(X, centroids)
    final_sse = float(np.sum(final.squared_distances, dtype=np.float64))
    final_time = perf_counter() - final_start
    compute_time = perf_counter() - compute_start
    metadata = dict(algorithm='fixed_minibatch_exact_cpu', config=asdict(config),
                    final_exact_sse=final_sse, steps=step,
                    termination_reason='tolerance' if converged else 'max_steps',
                    convergence_satisfied=converged,
                    stopping_rule='3 successive checkpoints: movement <= tolerance and unchanged probe labels',
                    stable_checks_required=_STABLE_CHECKS, stable_checks=stable_checks,
                    probe_size=len(probe), probe_checks=probe_checks,
                    probe_indices=probe_indices.tolist(), probe_role='overlapping training-population stopping diagnostic',
                    sampling_policy='uniform without replacement within batch; independent draws across steps; full batch uses input order',
                    sampling_rng='numpy.default_rng(seed)', probe_rng='numpy.default_rng(SeedSequence([seed,1]))',
                    cluster_observation_counts=counts.tolist(), observation_counts_include_C0=False,
                    points_processed=step * int(config.batch_size),
                    assignment_points_processed=step * int(config.batch_size) + (probe_checks+1)*len(probe) + n,
                    input_dtype='float32', centroid_dtype='float32', accumulation_dtype='float64',
                    empty_cluster_policy='retain', tie_policy='lowest_index', initialization='provided_C0',
                    history_objective_policy='batch/probe objectives only; final full SSE exclusively in final_exact_sse',
                    jit_policy='cold timings include CPU JIT; no speedup claim',
                    timing_scope='compute includes sampling, batch assignments, updates, monitoring and final assignment; component times are subsets, not additions to compute/wall')
    timings = dict(compute_s=compute_time, sampling_s=sampling_time,
                   batch_assignment_s=assignment_time, accumulation_update_s=update_time,
                   monitoring_s=monitoring_time, final_assignment_s=final_time,
                   wall_s=perf_counter()-started)
    return RunResult(centroids, final.labels, history, timings, metadata)
