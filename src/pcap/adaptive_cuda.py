"""Real CUDA adaptive batches; exact custom kernels, persistent device statistics."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .adaptive_controller import BatchPolicy, Checkpoint, initial_state, transition
from .contracts import AssignmentResult
from numba import cuda
from .minibatch_cuda import (_assign_kernel, _assign_indexed_kernel, _accumulate_indexed_kernel,
    _update_minibatch_kernel, _checkpoint_shift_kernel, _copy_centroids_kernel,
    _reduce, _blocks, _THREADS, _required_device_bytes)
from .contracts import FitConfig, RunResult, validate_points_centroids
from .minibatch import _sample_batch_indices


def fit_adaptive_minibatch_cuda(X, C0, config, *, policy=None) -> RunResult:
    """Persistent cumulative updates; controller runs only at probe checkpoints.

    Three stable adapting checkpoints request maximum-batch confirmation. Three
    fresh stable checkpoints there are required to stop. Adaptation stays frozen
    during confirmation; instability resets its counter, not its phase. Stability
    means absolute movement <= tolerance and unchanged exact probe labels, with
    no deterioration relative to the confirmation-entry objective. This remains
    a probe heuristic, not proof of full-data convergence. Explicit policy floors
    are honored; the default floor uses the initial probe RMS spread squared.
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
        raise ValueError('Adaptive exact CUDA Mini-Batch requires ann_effort=None')
    if policy is not None and not isinstance(policy, BatchPolicy):
        raise TypeError('policy must be BatchPolicy or None')
    supplied_policy = policy is not None
    policy = policy if supplied_policy else BatchPolicy.for_run(n, config.batch_size)
    if policy.initial_batch != config.batch_size or policy.maximum > n:
        raise ValueError('Policy must match initial batch and maximum <= N')
    if int(config.max_steps) * int(policy.maximum) > np.iinfo(np.int64).max:
        raise ValueError('Cumulative observation count exceeds int64 capacity')
    compute_start = perf_counter()
    rng = np.random.default_rng(config.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    backend = _DeviceUpdates(X, C0, int(policy.maximum), min(n, 1024))
    start = perf_counter()
    probe_indices = np.sort(probe_rng.choice(n, size=min(n, 1024), replace=False))
    probe = np.ascontiguousarray(X[probe_indices])
    probe64 = probe.astype(np.float64)
    spread_squared = float(np.mean(np.sum((probe64 - probe64.mean(axis=0))**2, axis=1)))
    scale = max(float(np.sqrt(spread_squared)), 1e-12)
    if not supplied_policy:
        policy = replace(policy, objective_floor=1e-12 * scale * scale)
    backend.set_probe(probe_indices)
    previous_probe = backend.probe()
    initial_objective = float(np.mean(previous_probe.squared_distances, dtype=np.float64))
    monitoring_time = perf_counter() - start
    start = perf_counter()
    state = initial_state(policy, initial_objective)
    controller_time = perf_counter() - start
    history = []
    sampling_time = assignment_time = update_time = 0.0
    stable_checks = confirmation_checks = probe_checks = controller_calls = 0
    observations = 0
    confirming = converged = False
    confirmation_objective = None
    confirmation_start = None
    for step in range(1, int(config.max_steps) + 1):
        batch_size = int(state.batch_size)
        phase = 'confirmation' if confirming else 'adapting'
        start = perf_counter()
        indices = _sample_batch_indices(rng, n, batch_size)
        sampling_time += perf_counter() - start
        start = perf_counter()
        assignment = backend.assign(indices)
        assignment_time += perf_counter() - start
        start = perf_counter()
        movement = backend.update()
        observations += batch_size
        update_time += perf_counter() - start
        entry = dict(step=step, phase=phase, batch_size=batch_size,
                     next_batch_size=batch_size, points_processed=observations,
                     cumulative_observations=observations, centroid_displacement=movement,
                     batch_sse_before_update=float(np.sum(assignment.squared_distances, dtype=np.float64)),
                     batch_objective_scope='sample only, at pre-update centroids',
                     convergence_checked=False, controller_decision=None)
        if step % int(config.probe_interval) == 0:
            start = perf_counter()
            current_probe = backend.probe()
            probe_sse = float(np.sum(current_probe.squared_distances, dtype=np.float64))
            objective = probe_sse / len(probe)
            displacement = backend.checkpoint_movement
            changes = float(np.mean(current_probe.labels != previous_probe.labels))
            checkpoint = Checkpoint(objective, displacement / scale, changes)
            stable = displacement <= config.tolerance and changes == 0
            probe_checks += 1
            previous_probe = current_probe
            monitoring_time += perf_counter() - start
            entry.update(convergence_checked=True, checkpoint_displacement=displacement,
                         normalized_checkpoint_displacement=checkpoint.movement,
                         probe_sse=probe_sse, probe_mean_squared_distance=objective,
                         probe_objective_scope='fixed overlapping training probe, post-update centroids',
                         probe_label_change_fraction=changes)
            start = perf_counter()
            if confirming:
                # A stopping-only quality gate, not an adaptive effort decision.
                quality_ok = objective <= confirmation_objective + policy.deterioration * max(
                    confirmation_objective, policy.objective_floor)
                confirmation_checks = confirmation_checks + 1 if stable and quality_ok else 0
                converged = confirmation_checks >= 3
                entry.update(adaptation_frozen=True, confirmation_quality_satisfied=quality_ok)
            else:
                before_state = state
                state, decision = transition(policy, state, checkpoint)
                controller_calls += 1
                stable_checks = stable_checks + 1 if stable and not decision.changed and not decision.deteriorated else 0
                entry.update(controller_decision={**asdict(decision), 'changed': decision.changed},
                             controller_state_before=asdict(before_state), controller_state_after=asdict(state),
                             adaptation_frozen=False)
                if stable_checks >= 3:
                    confirming = True
                    confirmation_objective = objective
                    confirmation_start = step
                    # Confirmation is execution state, not a controller-policy change.
                    state = replace(state, batch_size=int(policy.maximum), quiet_count=0, recent_changes=())
                    confirmation_checks = 0
                    entry['confirmation_entered'] = True
            entry.update(next_batch_size=int(state.batch_size), stable_checks=stable_checks,
                         confirmation_checks=confirmation_checks, convergence_satisfied=converged)
            controller_time += perf_counter() - start
        history.append(entry)
        if converged:
            break
    final_start = perf_counter()
    centroids, final, counts, final_sse = backend.final()
    final_time = perf_counter() - final_start
    compute_time = perf_counter() - compute_start
    metadata = dict(algorithm='adaptive_minibatch_exact_cuda', execution_device='cuda', gpu=str(cuda.get_current_device().name), config=asdict(config), policy=asdict(policy),
                    final_exact_sse=final_sse, steps=step,
                    termination_reason='tolerance' if converged else 'max_steps', convergence_satisfied=converged,
                    stopping_rule='3 stable adapting checkpoints then 3 fresh stable maximum-batch confirmation checkpoints',
                    confirmation_start_step=confirmation_start, confirmation_checks=confirmation_checks,
                    confirmation_active=confirming, stable_checks=stable_checks,
                    confirmation_objective=confirmation_objective,
                    confirmation_policy='frozen until termination; instability resets confirmation count; entry-objective quality gate',
                    controller_calls=controller_calls, controller_final_state=asdict(state),
                    probe_size=len(probe), probe_indices=probe_indices.tolist(), probe_checks=probe_checks,
                    probe_scale=scale, initial_probe_objective=initial_objective,
                    probe_role='overlapping training-population diagnostic, not held-out validation',
                    points_processed=observations, assignment_points_processed=observations+(probe_checks+1)*len(probe)+n,
                    cluster_observation_counts=counts.tolist(), observation_counts_include_C0=False,
                    sampling_policy='uniform without replacement within batch; independent draws across steps; full batch uses input order',
                    sampling_rng='numpy.default_rng(seed)', probe_rng='numpy.default_rng(SeedSequence([seed,1]))',
                    empty_cluster_policy='retain', tie_policy='lowest_index', initialization='provided_C0',
                    input_dtype='float32', centroid_dtype='float32', accumulation_dtype='float64',
                    history_objective_policy='batch/probe objectives only; final exact full SSE in final_exact_sse',
                    jit_policy='cold timings include CUDA compilation; no speedup claim',
                    timing_scope='components are disjoint subsets of compute_s, inside wall_s; never add components again')
    timings = dict(sampling_s=sampling_time, batch_assignment_s=assignment_time,
                   accumulation_update_s=update_time, monitoring_s=monitoring_time,
                   controller_s=controller_time, final_assignment_s=final_time,
                   compute_s=compute_time, wall_s=perf_counter()-started)
    metadata['estimated_device_buffer_bytes'] = backend.estimated_bytes
    metadata['transfer_policy'] = 'X uploaded once; per-update indices/scalars; probe labels/distances; final results downloaded once'
    return RunResult(centroids, final.labels, history, timings, metadata)


class _DeviceUpdates:
    """GPU-owned sums/counts/centroids. Host never updates individual points."""
    def __init__(self, X, C0, capacity, probe_size):
        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError('Real CUDA hardware required; use the WSL CUDA launcher')
        if cuda.get_current_device().compute_capability < (6, 0):
            raise RuntimeError('float64 atomics require compute capability >= 6.0')
        n, d = X.shape
        k = len(C0)
        if n > np.iinfo(np.int32).max:
            raise ValueError('N exceeds int32 indexed capacity')
        self.estimated_bytes = _required_device_bytes(n, d, k, capacity, probe_size) + 8*probe_size
        free, _ = cuda.current_context().get_memory_info()
        if self.estimated_bytes > .8*free:
            raise MemoryError('Adaptive CUDA buffers exceed 80% of free VRAM')
        self.X = cuda.to_device(X)
        self.C = cuda.to_device(C0)
        self.check_C = cuda.to_device(C0)
        self.sums = cuda.to_device(np.zeros((k, d), np.float64))
        self.counts = cuda.to_device(np.zeros(k, np.int64))
        self.indices = cuda.device_array(capacity, np.int32)
        self.labels = cuda.device_array(capacity, np.int32)
        self.distances = cuda.device_array(capacity, np.float64)
        self.partial = cuda.device_array(max(_blocks(n), _blocks(capacity), _blocks(k), _blocks(probe_size)), np.float64)
        self.scalar = cuda.device_array(1, np.float64)
        self.shifts = cuda.device_array(k, np.float64)
        self.probe_indices = cuda.device_array(probe_size, np.int32)
        self.probe_labels = cuda.device_array(probe_size, np.int32)
        self.probe_distances = cuda.device_array(probe_size, np.float64)
        self.final_labels = cuda.device_array(n, np.int32)
        self.final_distances = cuda.device_array(n, np.float64)
        self.checkpoint_movement = 0.
        cuda.synchronize()

    def set_probe(self, indices):
        self.probe_indices.copy_to_device(np.asarray(indices, np.int32))
        cuda.synchronize()

    def reduce(self, values, maximum=False):
        _reduce(values, self.partial, self.scalar, maximum)
        value = float(self.scalar.copy_to_host()[0])
        cuda.synchronize()
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite CUDA reduction')
        return value

    def assign(self, indices):
        start = perf_counter()
        self.b = len(indices)
        self.active_indices = self.indices[:self.b]
        self.active_labels = self.labels[:self.b]
        self.active_distances = self.distances[:self.b]
        self.active_indices.copy_to_device(np.asarray(indices, np.int32))
        _assign_indexed_kernel[_blocks(self.b), _THREADS](self.X, self.active_indices, self.C, self.active_labels, self.active_distances)
        sse = self.reduce(self.active_distances)
        # The integration consumes only the objective, never batch labels on host.
        return AssignmentResult(None, np.array([sse], np.float64), perf_counter()-start)

    def update(self):
        _accumulate_indexed_kernel[_blocks(self.b), _THREADS](self.X, self.active_indices, self.active_labels, self.sums, self.counts)
        _update_minibatch_kernel[_blocks(len(self.C)), _THREADS](self.C, self.sums, self.counts, self.shifts)
        return float(np.sqrt(self.reduce(self.shifts, True)))

    def probe(self):
        size = len(self.probe_indices)
        _assign_indexed_kernel[_blocks(size), _THREADS](self.X, self.probe_indices, self.C, self.probe_labels, self.probe_distances)
        _checkpoint_shift_kernel[_blocks(len(self.C)), _THREADS](self.C, self.check_C, self.shifts)
        self.checkpoint_movement = float(np.sqrt(self.reduce(self.shifts, True)))
        _copy_centroids_kernel[_blocks(self.C.size), _THREADS](self.C, self.check_C)
        labels = self.probe_labels.copy_to_host()
        distances = self.probe_distances.copy_to_host()
        cuda.synchronize()
        return AssignmentResult(labels, distances, 0.)

    def final(self):
        _assign_kernel[_blocks(len(self.X)), _THREADS](self.X, self.C, self.final_labels, self.final_distances)
        sse = self.reduce(self.final_distances)
        C, labels, counts = self.C.copy_to_host(), self.final_labels.copy_to_host(), self.counts.copy_to_host()
        cuda.synchronize()
        if not np.isfinite(C).all():
            raise FloatingPointError('Nonfinite final CUDA centroids')
        return C, AssignmentResult(labels, None, 0.), counts, sse
