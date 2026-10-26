"""Experimental GPU Mini-Batch K-Means with decayed historical statistics.

This module is intentionally separate from the cumulative G1/G3CUDA baselines.
It keeps points, centroids, decayed float64 sums/effective counts, and temporary
batch statistics resident on the CUDA device. It never receives a frozen
quality target or exact-reference SSE.
"""

from dataclasses import asdict
import math
from time import perf_counter

import numpy as np
from numba import cuda, float64

from .adaptive_v2_controller import (
    AdaptiveV2Checkpoint,
    AdaptiveV2Policy,
    initial_v2_state,
    transition_v2,
)
from .contracts import FitConfig, RunResult, validate_points_centroids, validate_integer
from .minibatch import _PROBE_SIZE, _STABLE_CHECKS, _sample_batch_indices
from .minibatch_cuda import (
    _MAX_BLOCKS,
    _THREADS,
    _assign_indexed_kernel,
    _assign_kernel,
    _blocks,
    _checkpoint_shift_kernel,
    _copy_centroids_kernel,
    _reduce,
    _timed_kernels,
)


@cuda.jit
def _zero_batch_stats_kernel(batch_sums, batch_counts):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = batch_sums.size + batch_counts.size
    for pos in range(idx, total, stride):
        if pos < batch_sums.size:
            batch_sums[pos // batch_sums.shape[1], pos % batch_sums.shape[1]] = 0.0
        else:
            batch_counts[pos - batch_sums.size] = 0


@cuda.jit
def _accumulate_batch_kernel(points, indices, labels, batch_sums, batch_counts):
    for i in range(cuda.grid(1), indices.shape[0], cuda.gridsize(1)):
        point = indices[i]
        cluster = labels[i]
        cuda.atomic.add(batch_counts, cluster, 1)
        for feature in range(points.shape[1]):
            cuda.atomic.add(batch_sums, (cluster, feature), float64(points[point, feature]))


@cuda.jit
def _decayed_centroid_update_kernel(centroids, sums, effective_counts,
                                    batch_sums, batch_counts, decay, shifts):
    for cluster in range(cuda.grid(1), centroids.shape[0], cuda.gridsize(1)):
        weight = decay * effective_counts[cluster] + float64(batch_counts[cluster])
        shift = 0.0
        if weight > 0.0:
            for feature in range(centroids.shape[1]):
                total = decay * sums[cluster, feature] + batch_sums[cluster, feature]
                sums[cluster, feature] = total
                old = float64(centroids[cluster, feature])
                centroids[cluster, feature] = total / weight
                delta = float64(centroids[cluster, feature]) - old
                shift += delta * delta
        else:
            # A cluster without historical or current observations retains C.
            for feature in range(centroids.shape[1]):
                sums[cluster, feature] *= decay
        effective_counts[cluster] = weight
        shifts[cluster] = shift


def _validate_fit(X, C0, config, policy, batch_indices):
    if not isinstance(config, FitConfig):
        raise TypeError("config must be FitConfig")
    validate_points_centroids(X, C0)
    n, _ = X.shape
    if len(C0) != config.k or config.k > n:
        raise ValueError("C0 rows must equal config.k, with k <= N")
    if config.batch_size is None or config.batch_size > n:
        raise ValueError("batch_size must be supplied and <= N")
    if config.ann_effort is not None:
        raise ValueError("Decayed exact CUDA Mini-Batch requires ann_effort=None")
    if policy is not None:
        if not isinstance(policy, AdaptiveV2Policy):
            raise TypeError("policy must be AdaptiveV2Policy or None")
        if policy.initial_batch != config.batch_size or policy.maximum > n:
            raise ValueError("Policy initial batch must match config; maximum must be <= N")
    capacity = int(policy.maximum) if policy is not None else int(config.batch_size)
    if int(config.max_steps) * capacity > np.iinfo(np.int64).max:
        raise ValueError("Maximum processed-observation count exceeds int64 capacity")
    if n > np.iinfo(np.int32).max:
        raise ValueError("Indexed CUDA Mini-Batch requires N within int32 capacity")
    if batch_indices is not None:
        samples = np.asarray(batch_indices)
        if samples.ndim != 2 or samples.shape[0] != config.max_steps:
            raise ValueError("batch_indices must have shape [max_steps, batch_size]")
        if samples.shape[1] != config.batch_size or samples.dtype.kind not in "iu":
            raise ValueError("batch_indices must be integer with configured batch width")
        if samples.size and (int(samples.min()) < 0 or int(samples.max()) >= n):
            raise ValueError("batch_indices contain out-of-range point indices")
    return capacity


class _DecayedDeviceState:
    def __init__(self, X, C0, capacity, probe_indices):
        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError("Adaptive V2 requires real CUDA hardware; use scripts/run_cuda_tests.sh")
        device = cuda.get_current_device()
        if device.compute_capability < (6, 0):
            raise RuntimeError("float64 atomic accumulation requires compute capability >= 6.0")
        n, d = X.shape
        k = len(C0)
        if n > np.iinfo(np.int32).max:
            raise ValueError("N exceeds int32 indexed capacity")
        self.device = device
        self.n, self.d, self.k = n, d, k
        self.capacity = int(capacity)
        self.probe_size = len(probe_indices)
        self.estimated_bytes = (
            4*n*d + 4*k*d*2 + 8*k*d*2 + 8*k*2
            + 4*self.capacity*2 + 8*self.capacity
            + 4*self.probe_size*2 + 8*self.probe_size
            + 4*n + 8*n
            + 8*(_blocks(n)+_blocks(self.capacity)+_blocks(self.probe_size)+_blocks(k)+3)
        )
        free, self.total_bytes = cuda.current_context().get_memory_info()
        if self.estimated_bytes > 0.8 * free:
            raise MemoryError(f"Adaptive V2 buffers need {self.estimated_bytes} bytes; {free} free")
        self.X = cuda.to_device(X)
        self.C = cuda.to_device(C0)
        self.check_C = cuda.to_device(C0)
        self.sums = cuda.to_device(np.zeros((k, d), np.float64))
        self.weights = cuda.to_device(np.zeros(k, np.float64))
        self.batch_sums = cuda.device_array((k, d), np.float64)
        self.batch_counts = cuda.device_array(k, np.int64)
        self.indices = cuda.device_array(self.capacity, np.int32)
        self.labels = cuda.device_array(self.capacity, np.int32)
        self.distances = cuda.device_array(self.capacity, np.float64)
        self.probe_indices = cuda.to_device(np.asarray(probe_indices, np.int32))
        self.probe_labels = cuda.device_array(self.probe_size, np.int32)
        self.probe_distances = cuda.device_array(self.probe_size, np.float64)
        self.final_labels = cuda.device_array(n, np.int32)
        self.final_distances = cuda.device_array(n, np.float64)
        self.shifts = cuda.device_array(k, np.float64)
        max_partials = max(_blocks(n), _blocks(self.capacity), _blocks(self.probe_size), _blocks(k))
        self.partials = cuda.device_array(max_partials, np.float64)
        self.scalar = cuda.device_array(1, np.float64)
        cuda.synchronize()

    def _scalar(self, values, maximum=False):
        blocks = _blocks(values.size)
        _reduce(values, self.partials[:blocks], self.scalar, maximum)
        value = float(self.scalar.copy_to_host()[0])
        cuda.synchronize()
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite CUDA reduction")
        return value

    def assign(self, indices):
        b = len(indices)
        d_idx, d_lab, d_dst = self.indices[:b], self.labels[:b], self.distances[:b]
        d_idx.copy_to_device(np.ascontiguousarray(indices, dtype=np.int32))
        _assign_indexed_kernel[_blocks(b), _THREADS](self.X, d_idx, self.C, d_lab, d_dst)
        return self._scalar(d_dst), b

    def update(self, b, decay):
        _zero_batch_stats_kernel[_blocks(self.k*self.d+self.k), _THREADS](self.batch_sums, self.batch_counts)
        _accumulate_batch_kernel[_blocks(b), _THREADS](
            self.X, self.indices[:b], self.labels[:b], self.batch_sums, self.batch_counts)
        _decayed_centroid_update_kernel[_blocks(self.k), _THREADS](
            self.C, self.sums, self.weights, self.batch_sums, self.batch_counts,
            decay, self.shifts)
        return math.sqrt(self._scalar(self.shifts, maximum=True))

    def probe(self):
        _assign_indexed_kernel[_blocks(self.probe_size), _THREADS](
            self.X, self.probe_indices, self.C, self.probe_labels, self.probe_distances)
        sse = self._scalar(self.probe_distances)
        _checkpoint_shift_kernel[_blocks(self.k), _THREADS](self.C, self.check_C, self.shifts)
        movement = math.sqrt(self._scalar(self.shifts, maximum=True))
        _copy_centroids_kernel[_blocks(self.C.size), _THREADS](self.C, self.check_C)
        labels = self.probe_labels.copy_to_host()
        cuda.synchronize()
        return labels, sse, movement

    def quality(self):
        _assign_kernel[_blocks(self.n), _THREADS](
            self.X, self.C, self.final_labels, self.final_distances)
        return self._scalar(self.final_distances)

    def final(self):
        sse = self.quality()
        centroids = self.C.copy_to_host()
        labels = self.final_labels.copy_to_host()
        weights = self.weights.copy_to_host()
        sums = self.sums.copy_to_host()
        cuda.synchronize()
        if not (np.isfinite(centroids).all() and np.isfinite(weights).all() and np.isfinite(sums).all()):
            raise FloatingPointError("Nonfinite decayed CUDA state")
        return centroids, np.ascontiguousarray(labels, dtype=np.int32), weights, sse


def fit_decayed_minibatch_cuda(X, C0, config, *, quality_interval=100,
                               batch_indices=None) -> RunResult:
    """Fixed-batch experimental decayed-count CUDA Mini-Batch fit.

    ``batch_indices`` optionally supplies a reproducible replay stream shaped
    ``[max_steps, batch_size]``. Quality checks never influence fitting.
    """
    return _fit(X, C0, config, policy=None, adaptive=False,
                quality_interval=quality_interval, batch_indices=batch_indices)


def fit_adaptive_decayed_minibatch_cuda(X, C0, config, *, policy=None,
                                        quality_interval=100) -> RunResult:
    """Experimental Adaptive V2; the controller sees probe signals only."""
    return _fit(X, C0, config, policy=policy, adaptive=True,
                quality_interval=quality_interval, batch_indices=None)


def _fit(X, C0, config, *, policy, adaptive, quality_interval, batch_indices):
    wall_start = perf_counter()
    if adaptive and policy is None:
        if not isinstance(config, FitConfig):
            raise TypeError("config must be FitConfig")
        if config.batch_size is None:
            raise ValueError("batch_size must be supplied for adaptive Mini-Batch")
        policy = AdaptiveV2Policy.for_run(len(X), int(config.batch_size))
    capacity = _validate_fit(X, C0, config, policy, batch_indices)
    validate_integer("quality_interval", quality_interval)
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Adaptive V2 requires real CUDA; simulator or CPU fallback is unsupported")
    n, _ = X.shape
    probe_size = min(n, _PROBE_SIZE)
    rng = np.random.default_rng(config.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    probe_indices = np.sort(probe_rng.choice(n, size=probe_size, replace=False)).astype(np.int32)
    probe = np.asarray(X[probe_indices], dtype=np.float64)
    centered = probe - probe.mean(axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum(centered*centered, axis=1)))), 1e-12)
    if adaptive:
        policy = policy or AdaptiveV2Policy.for_run(n, config.batch_size)
        if not isinstance(policy, AdaptiveV2Policy):
            raise TypeError("policy must be AdaptiveV2Policy")
        if policy.initial_batch != config.batch_size or policy.maximum > n:
            raise ValueError("Policy initial batch must match config and maximum <= N")
    state = _DecayedDeviceState(X, C0, capacity, probe_indices)
    rng_name = "provided pre-generated batch-index stream" if batch_indices is not None else "numpy.default_rng(config.seed)"
    prior_labels, initial_probe_sse, _ = state.probe()
    controller_state = initial_v2_state(policy, initial_probe_sse / probe_size) if adaptive else None
    stable_checks = confirmation_checks = probe_checks = controller_calls = 0
    confirming = converged = False
    confirmation_objective = None
    history, quality_history = [], []
    observations = 0
    sampling_s = assignment_s = update_s = monitoring_s = controller_s = quality_s = 0.0

    def record_quality(step):
        started = perf_counter()
        objective = state.quality()
        elapsed = perf_counter() - started
        quality_history.append(dict(step=step, elapsed_s=perf_counter()-wall_start,
                                   full_data_sse=objective, points_processed=observations))
        return elapsed, objective

    if quality_interval:
        e, _ = record_quality(0)
        quality_s += e
    total = int(config.max_steps)
    if batch_indices is not None:
        batch_indices = np.asarray(batch_indices)
    batch_rng = rng
    previous_probe_sse = initial_probe_sse
    for step in range(1, total+1):
        batch = int(controller_state.batch_size) if adaptive else int(config.batch_size)
        phase = "confirmation" if confirming else "adapting" if adaptive else "fixed"
        start = perf_counter()
        if batch_indices is None:
            sampled = _sample_batch_indices(batch_rng, n, batch)
        else:
            if batch != int(config.batch_size):
                raise ValueError("batch-index replay is supported only for fixed-batch fits")
            sampled = batch_indices[step-1]
        sampling_s += perf_counter()-start
        start = perf_counter()
        batch_sse, active_batch = state.assign(sampled)
        assignment_s += perf_counter()-start
        decay = 0.5 ** (active_batch / (10.0 * n))
        start = perf_counter()
        movement = state.update(active_batch, decay)
        update_s += perf_counter()-start
        observations += active_batch
        row = dict(step=step, phase=phase, batch_size=active_batch,
                   next_batch_size=batch, points_processed=observations,
                   centroid_displacement=movement, forgetting_factor=decay,
                   batch_sse_before_update=batch_sse,
                   objective_scope="sample only, assigned at pre-update centroids",
                   controller_decision=None, convergence_checked=False)
        if step % int(config.probe_interval) == 0:
            start = perf_counter()
            current_labels, probe_sse, displacement = state.probe()
            label_change = float(np.mean(current_labels != prior_labels))
            prior_labels = current_labels
            relative_movement = displacement / scale
            probe_mean = probe_sse / probe_size
            stable = displacement <= config.tolerance and label_change == 0.0
            probe_checks += 1
            row.update(convergence_checked=True, probe_sse=probe_sse,
                       probe_mean_squared_distance=probe_mean,
                       probe_objective_scope="fixed overlapping training-population probe",
                       normalized_checkpoint_displacement=relative_movement,
                       probe_label_change_fraction=label_change)
            monitoring_s += perf_counter()-start
            start = perf_counter()
            if adaptive and not confirming:
                before = controller_state
                controller_state, decision = transition_v2(
                    policy, controller_state,
                    AdaptiveV2Checkpoint(probe_mean, relative_movement, label_change))
                controller_calls += 1
                stable_checks = stable_checks+1 if stable and not decision.changed and not decision.deteriorated else 0
                decision_log = asdict(decision)
                decision_log["changed"] = decision.changed
                row.update(controller_decision=decision_log,
                           controller_state_before=asdict(before),
                           controller_state_after=asdict(controller_state), adaptation_frozen=False)
                if stable_checks >= 3:
                    confirming = True
                    confirmation_objective = probe_mean
                    confirmation_checks = 0
                    controller_state = type(controller_state)(
                        batch_size=policy.maximum,
                        previous_probe_sse=controller_state.previous_probe_sse,
                        best_probe_sse=controller_state.best_probe_sse,
                        safe_batch_size=controller_state.safe_batch_size,
                        stagnation_count=0, progress_count=0, recent_changes=(),
                        checkpoints=controller_state.checkpoints)
                    row["confirmation_entered"] = True
            elif confirming:
                allowed = confirmation_objective * 1.01 + policy.objective_floor
                confirmation_checks = confirmation_checks + 1 if stable and probe_mean <= allowed else 0
                converged = confirmation_checks >= 3
                row.update(adaptation_frozen=True,
                           confirmation_quality_satisfied=probe_mean <= allowed)
            else:
                stable_checks = stable_checks + 1 if stable else 0
                converged = stable_checks >= _STABLE_CHECKS
                row.update(adaptation_frozen=True)
            row.update(next_batch_size=(controller_state.batch_size if adaptive else active_batch),
                       stable_checks=stable_checks, confirmation_checks=confirmation_checks,
                       convergence_satisfied=converged)
            controller_s += perf_counter()-start
        history.append(row)
        if quality_interval and step % int(quality_interval) == 0:
            elapsed, _ = record_quality(step)
            quality_s += elapsed
        if converged:
            break
    if not quality_history or quality_history[-1]["step"] != step:
        elapsed, _ = record_quality(step)
        quality_s += elapsed
    final_start = perf_counter()
    centroids, labels, weights, final_sse = state.final()
    final_assignment_s = perf_counter()-final_start
    if quality_history and quality_history[-1]["step"] == step:
        quality_history[-1].update(full_data_sse=final_sse, final_output=True,
                                   points_processed=observations)
    metadata = dict(
        algorithm="adaptive_v2_decayed_cuda" if adaptive else "fixed_decayed_cuda",
        execution_device="cuda", gpu=str(state.device.name),
        config=asdict(config), policy=asdict(policy) if adaptive else None,
        final_exact_sse=final_sse, steps=step,
        termination_reason="tolerance" if converged else "max_steps",
        convergence_satisfied=converged,
        stopping_rule="three stable adapting probes then three fresh frozen maximum-batch checks" if adaptive else "three stable fixed-probe checks",
        probe_indices=probe_indices.tolist(), probe_size=probe_size,
        probe_checks=probe_checks, controller_calls=controller_calls,
        controller_final_state=asdict(controller_state) if adaptive else None,
        stable_checks=stable_checks, confirmation_checks=confirmation_checks,
        confirmation_objective=confirmation_objective,
        points_processed=observations, effective_cluster_counts=weights.tolist(),
        observation_counts_include_C0=False,
        sampling_policy="uniform without replacement per batch; independent draws across steps" if batch_indices is None else "caller-supplied replay indices",
        sampling_rng=rng_name, probe_rng="numpy.default_rng(SeedSequence([seed,1]))",
        decay_rule="lambda_t=0.5**(actual_batch_size/(10*N)); S=lambda_t*S+batch_sum; W=lambda_t*W+batch_count",
        decay_half_life_passes=10,
        empty_cluster_policy="retain previous centroid when effective count is zero",
        tie_policy="lowest_index", initialization="provided_C0",
        centroid_dtype="float32", accumulation_dtype="float64",
        quality_history=quality_history, quality_evaluation_points=len(quality_history)*n,
        controller_signals="fixed-probe mean squared distance, relative probe trend, normalized centroid movement and probe label changes; no quality target/reference supplied",
        history_objective_policy="batch/probe metrics separate; exact full-data SSE in quality_history and final_exact_sse",
        estimated_device_buffer_bytes=state.estimated_bytes,
        timing_scope="host wall and component scopes; full-data quality evaluations included; components are subsets, do not add to wall_s",
    )
    timings = dict(sampling_s=sampling_s, batch_assignment_s=assignment_s,
                   accumulation_update_s=update_s, monitoring_s=monitoring_s,
                   controller_s=controller_s, quality_evaluation_s=quality_s,
                   final_assignment_s=final_assignment_s,
                   compute_s=sampling_s+assignment_s+update_s+monitoring_s+controller_s+quality_s+final_assignment_s,
                   wall_s=perf_counter()-wall_start)
    return RunResult(centroids, labels, history, timings, metadata)
