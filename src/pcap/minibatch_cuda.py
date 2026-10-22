"""Fixed Mini-Batch K-Means with custom Numba-CUDA kernels.

Import this optional module only when CUDA is needed. Shared contracts and CPU
modules never import it. Mini-batch assignment, persistent accumulation and
centroid updates run on the GPU; host loops schedule steps, manage sampling,
and track convergence. Input arrays and persistent cluster statistics stay on
device throughout fitting.
"""

from dataclasses import asdict
from importlib.metadata import version
import math
import platform
from time import perf_counter

import numpy as np
from numba import cuda, float64
from numba.cuda.cudadrv import driver, runtime

from .contracts import FitConfig, RunResult, validate_points_centroids
from .minibatch import _PROBE_SIZE, _STABLE_CHECKS, _movement, _sample_batch_indices

_THREADS = 256
_MAX_BLOCKS = 1024


def _blocks(size):
    return min(_MAX_BLOCKS, (size + _THREADS - 1) // _THREADS)


@cuda.jit
def _assign_kernel(points, centroids, labels, distances):
    """Grid-stride nearest-centroid assignment for all points."""
    for i in range(cuda.grid(1), points.shape[0], cuda.gridsize(1)):
        best_distance = math.inf
        best_label = 0
        for j in range(centroids.shape[0]):
            distance = 0.0
            for d in range(points.shape[1]):
                delta = float64(points[i, d]) - float64(centroids[j, d])
                distance += delta * delta
            if distance < best_distance:
                best_distance = distance
                best_label = j
        labels[i] = best_label
        distances[i] = best_distance


@cuda.jit
def _assign_indexed_kernel(points, indices, centroids, labels, distances):
    """Nearest-centroid assignment for an indexed subset (batch or probe)."""
    for i in range(cuda.grid(1), indices.shape[0], cuda.gridsize(1)):
        idx = indices[i]
        best_distance = math.inf
        best_label = 0
        for j in range(centroids.shape[0]):
            distance = 0.0
            for d in range(points.shape[1]):
                delta = float64(points[idx, d]) - float64(centroids[j, d])
                distance += delta * delta
            if distance < best_distance:
                best_distance = distance
                best_label = j
        labels[i] = best_label
        distances[i] = best_distance


@cuda.jit
def _accumulate_indexed_kernel(points, indices, labels, sums, counts):
    """Atomic accumulation into persistent cluster sums and observation counts."""
    for i in range(cuda.grid(1), indices.shape[0], cuda.gridsize(1)):
        idx = indices[i]
        label = labels[i]
        cuda.atomic.add(counts, label, 1)
        for d in range(points.shape[1]):
            cuda.atomic.add(sums, (label, d), float64(points[idx, d]))


@cuda.jit
def _update_minibatch_kernel(centroids, sums, counts, squared_shifts):
    """Incremental centroid update: C_j = S_j / N_j for active clusters."""
    for j in range(cuda.grid(1), centroids.shape[0], cuda.gridsize(1)):
        shift = 0.0
        if counts[j] > 0:
            for d in range(centroids.shape[1]):
                old_value = float64(centroids[j, d])
                centroids[j, d] = sums[j, d] / counts[j]
                delta = float64(centroids[j, d]) - old_value
                shift += delta * delta
        squared_shifts[j] = shift


@cuda.jit
def _checkpoint_shift_kernel(centroids, check_centroids, squared_shifts):
    """Compute squared centroid displacement relative to a checkpoint."""
    for j in range(cuda.grid(1), centroids.shape[0], cuda.gridsize(1)):
        shift = 0.0
        for d in range(centroids.shape[1]):
            delta = float64(centroids[j, d]) - float64(check_centroids[j, d])
            shift += delta * delta
        squared_shifts[j] = shift


@cuda.jit
def _copy_centroids_kernel(src, dst):
    """Copy centroids array on device."""
    for i in range(cuda.grid(1), src.size, cuda.gridsize(1)):
        r = i // src.shape[1]
        c = i % src.shape[1]
        dst[r, c] = src[r, c]


@cuda.jit
def _reduce_kernel(values, partials, use_max):
    """Fixed-size block reduction for SSE or max displacement."""
    shared = cuda.shared.array(_THREADS, dtype=float64)
    thread = cuda.threadIdx.x
    value = 0.0
    for i in range(cuda.grid(1), values.size, cuda.gridsize(1)):
        if use_max:
            value = max(value, values[i])
        else:
            value += values[i]
    shared[thread] = value
    cuda.syncthreads()
    stride = _THREADS // 2
    while stride > 0:
        if thread < stride:
            if use_max:
                shared[thread] = max(shared[thread], shared[thread + stride])
            else:
                shared[thread] += shared[thread + stride]
        cuda.syncthreads()
        stride //= 2
    if thread == 0:
        partials[cuda.blockIdx.x] = shared[0]


def _reduce(values, partials, scalar, use_max):
    _reduce_kernel[partials.size, _THREADS](values, partials, use_max)
    _reduce_kernel[1, _THREADS](partials, scalar, use_max)


def _timed_kernels(action):
    """Return synchronized host duration and CUDA-event interval in seconds."""
    cuda.synchronize()
    start_event, end_event = cuda.event(timing=True), cuda.event(timing=True)
    start = perf_counter()
    start_event.record()
    action()
    end_event.record()
    end_event.synchronize()
    return perf_counter() - start, start_event.elapsed_time(end_event) / 1000.0


def _required_device_bytes(n, d, k, batch_size, probe_size):
    # X, centroids, check_centroids, sums, counts, shifts,
    # batch (indices, labels, distances, partials),
    # probe (indices, labels, distances, partials),
    # final pass (labels, distances, partials), scalars.
    return (
        4 * n * d                  # X (float32)
        + 4 * k * d * 2            # centroids, check_centroids (float32)
        + 8 * k * d                # sums (float64)
        + 8 * k                    # counts (int64)
        + 8 * k                    # shifts (float64)
        + 4 * batch_size           # batch_indices (int32)
        + 4 * batch_size           # batch_labels (int32)
        + 8 * batch_size           # batch_distances (float64)
        + 4 * probe_size           # probe_indices (int32)
        + 4 * probe_size           # probe_labels (int32)
        + 8 * probe_size           # probe_distances (float64)
        + 4 * n                    # final labels (int32)
        + 8 * n                    # final distances (float64)
        + 8 * (_blocks(n) + _blocks(batch_size) + _blocks(probe_size) + _blocks(k) + 3)
    )


def fit_minibatch_cuda(X, C0, config) -> RunResult:
    """Fit fixed Mini-Batch K-Means on the current visible CUDA device.

    Persistent cumulative updates keep float64 sums and int64 observation counts
    on device across batches. Mini-batch assignment, incremental accumulation and
    centroid updates are executed via custom CUDA kernels. Stopping is evaluated
    every config.probe_interval steps on a fixed seeded probe. Three successive
    checks must satisfy checkpoint displacement <= config.tolerance AND unchanged
    probe labels. A final exact full-data pass evaluates final labels and SSE.
    """
    wall_start = perf_counter()
    if not isinstance(config, FitConfig):
        raise TypeError("config must be FitConfig")
    validate_points_centroids(X, C0)
    n, d = X.shape
    k = config.k
    if len(C0) != k or k > n:
        raise ValueError("C0 rows must equal config.k, with k <= N")
    if config.batch_size is None or config.batch_size > n:
        raise ValueError("batch_size must be supplied and <= N")
    if config.ann_effort is not None:
        raise ValueError("Fixed exact CUDA Mini-Batch requires ann_effort=None")
    if int(config.max_steps) * int(config.batch_size) > np.iinfo(np.int64).max:
        raise ValueError("Cumulative observation count exceeds int64 capacity")
    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError("fit_minibatch_cuda requires real hardware, not CUDA simulator mode")
    if not cuda.is_available():
        raise RuntimeError("CUDA unavailable; on WSL2 use scripts/run_cuda_tests.sh for driver discovery")
    device = cuda.get_current_device()
    if device.compute_capability < (6, 0):
        raise RuntimeError("float64 atomic accumulation requires compute capability >= 6.0")

    batch_size = int(config.batch_size)
    probe_size = min(n, _PROBE_SIZE)
    estimated_bytes = _required_device_bytes(n, d, k, batch_size, probe_size)
    free_bytes, total_bytes = cuda.current_context().get_memory_info()
    if estimated_bytes > 0.8 * free_bytes:
        raise MemoryError(
            f"Resident CUDA buffers require {estimated_bytes} bytes; "
            f"free device memory is {free_bytes}. Use a smaller batch size or crop."
        )

    setup_start = perf_counter()
    upload_start = perf_counter()
    d_points = cuda.to_device(X)
    d_centroids = cuda.to_device(C0)
    d_check_centroids = cuda.to_device(C0)
    cuda.synchronize()
    upload_time = perf_counter() - upload_start

    # Persistent statistics on device
    d_sums = cuda.to_device(np.zeros((k, d), dtype=np.float64))
    d_counts = cuda.to_device(np.zeros(k, dtype=np.int64))
    d_shifts = cuda.device_array(k, dtype=np.float64)

    # Batch buffers
    d_batch_indices = cuda.device_array(batch_size, dtype=np.int32)
    d_batch_labels = cuda.device_array(batch_size, dtype=np.int32)
    d_batch_distances = cuda.device_array(batch_size, dtype=np.float64)
    d_batch_sse_partials = cuda.device_array(_blocks(batch_size), dtype=np.float64)

    # Probe buffers
    d_probe_indices = cuda.device_array(probe_size, dtype=np.int32)
    d_probe_labels = cuda.device_array(probe_size, dtype=np.int32)
    d_probe_distances = cuda.device_array(probe_size, dtype=np.float64)
    d_probe_sse_partials = cuda.device_array(_blocks(probe_size), dtype=np.float64)

    # Reduction scalars
    d_shift_partials = cuda.device_array(_blocks(k), dtype=np.float64)
    d_shift_scalar = cuda.device_array(1, dtype=np.float64)
    d_sse_scalar = cuda.device_array(1, dtype=np.float64)

    # Final pass buffers (allocated upfront)
    d_final_labels = cuda.device_array(n, dtype=np.int32)
    d_final_distances = cuda.device_array(n, dtype=np.float64)
    d_final_sse_partials = cuda.device_array(_blocks(n), dtype=np.float64)

    # Seeded RNG streams matching CPU reference
    rng = np.random.default_rng(config.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    probe_indices = np.sort(probe_rng.choice(n, size=probe_size, replace=False)).astype(np.int32)
    cuda.to_device(probe_indices, to=d_probe_indices)
    cuda.synchronize()
    setup_time = perf_counter() - setup_start

    kernel_time = 0.0
    scalar_transfer_time = 0.0
    sampling_time = 0.0
    assignment_time = 0.0
    update_time = 0.0
    monitoring_time = 0.0

    def read_scalar(buffer):
        nonlocal scalar_transfer_time
        start = perf_counter()
        val = float(buffer.copy_to_host()[0])
        cuda.synchronize()
        scalar_transfer_time += perf_counter() - start
        return val

    # Initial probe evaluation
    m_start = perf_counter()
    def eval_initial_probe():
        _assign_indexed_kernel[_blocks(probe_size), _THREADS](
            d_points, d_probe_indices, d_centroids, d_probe_labels, d_probe_distances)
    _, el = _timed_kernels(eval_initial_probe)
    kernel_time += el
    previous_probe_labels = d_probe_labels.copy_to_host()
    cuda.synchronize()
    monitoring_time += perf_counter() - m_start

    history = []
    stable_checks = probe_checks = 0
    converged = False
    compute_start = perf_counter()

    for step in range(1, int(config.max_steps) + 1):
        # 1. Sample batch indices
        s_start = perf_counter()
        raw_indices = _sample_batch_indices(rng, n, batch_size)
        batch_indices = np.ascontiguousarray(raw_indices, dtype=np.int32)
        cuda.to_device(batch_indices, to=d_batch_indices)
        cuda.synchronize()
        sampling_time += perf_counter() - s_start

        # 2. Batch assignment (against pre-update centroids)
        a_start = perf_counter()
        def assign_batch():
            _assign_indexed_kernel[_blocks(batch_size), _THREADS](
                d_points, d_batch_indices, d_centroids, d_batch_labels, d_batch_distances)
            _reduce(d_batch_distances, d_batch_sse_partials, d_sse_scalar, False)
        h_el, k_el = _timed_kernels(assign_batch)
        kernel_time += k_el
        batch_sse = read_scalar(d_sse_scalar)
        assignment_time += perf_counter() - a_start

        # 3. Accumulate batch statistics and update centroids
        u_start = perf_counter()
        def update_batch():
            _accumulate_indexed_kernel[_blocks(batch_size), _THREADS](
                d_points, d_batch_indices, d_batch_labels, d_sums, d_counts)
            _update_minibatch_kernel[_blocks(k), _THREADS](
                d_centroids, d_sums, d_counts, d_shifts)
            _reduce(d_shifts, d_shift_partials, d_shift_scalar, True)
        h_el, k_el = _timed_kernels(update_batch)
        kernel_time += k_el
        movement = math.sqrt(read_scalar(d_shift_scalar))
        update_time += perf_counter() - u_start

        entry = dict(
            step=step, batch_size=batch_size,
            points_processed=step * batch_size,
            centroid_displacement=movement,
            batch_sse_before_update=batch_sse,
            batch_objective_scope="sample only, at pre-update centroids",
            cumulative_observations=step * batch_size,
            convergence_checked=False,
        )

        # 4. Checkpoint monitoring
        if step % config.probe_interval == 0:
            m_start = perf_counter()
            def eval_probe():
                _assign_indexed_kernel[_blocks(probe_size), _THREADS](
                    d_points, d_probe_indices, d_centroids, d_probe_labels, d_probe_distances)
                _reduce(d_probe_distances, d_probe_sse_partials, d_sse_scalar, False)
                _checkpoint_shift_kernel[_blocks(k), _THREADS](
                    d_centroids, d_check_centroids, d_shifts)
                _reduce(d_shifts, d_shift_partials, d_shift_scalar, True)
            h_el, k_el = _timed_kernels(eval_probe)
            kernel_time += k_el
            probe_sse = read_scalar(d_sse_scalar)
            checkpoint_movement = math.sqrt(read_scalar(d_shift_scalar))

            current_probe_labels = d_probe_labels.copy_to_host()
            cuda.synchronize()
            changes = float(np.mean(current_probe_labels != previous_probe_labels))
            stable = checkpoint_movement <= config.tolerance and changes == 0.0
            stable_checks = stable_checks + 1 if stable else 0
            converged = stable_checks >= _STABLE_CHECKS
            probe_checks += 1

            entry.update(
                convergence_checked=True,
                checkpoint_displacement=checkpoint_movement,
                probe_sse=probe_sse,
                probe_objective_scope="fixed stopping probe only, at post-update centroids",
                probe_label_change_fraction=changes,
                stable_checks=stable_checks,
                convergence_satisfied=converged,
            )

            # Update check_centroids on device
            def copy_check():
                _copy_centroids_kernel[_blocks(k * d), _THREADS](d_centroids, d_check_centroids)
            _, k_el = _timed_kernels(copy_check)
            kernel_time += k_el
            previous_probe_labels = current_probe_labels
            monitoring_time += perf_counter() - m_start

        history.append(entry)
        if converged:
            break

    # 5. Final full-data exact evaluation pass
    f_start = perf_counter()
    def eval_final():
        _assign_kernel[_blocks(n), _THREADS](
            d_points, d_centroids, d_final_labels, d_final_distances)
        _reduce(d_final_distances, d_final_sse_partials, d_sse_scalar, False)
    h_el, k_el = _timed_kernels(eval_final)
    kernel_time += k_el
    final_sse = read_scalar(d_sse_scalar)
    final_assignment_time = perf_counter() - f_start
    cuda.synchronize()
    compute_time = perf_counter() - compute_start

    # Download results to host
    download_start = perf_counter()
    host_centroids = d_centroids.copy_to_host()
    host_labels = d_final_labels.copy_to_host()
    host_counts = d_counts.copy_to_host()
    cuda.synchronize()
    download_time = perf_counter() - download_start

    if not np.isfinite(host_centroids).all():
        raise FloatingPointError("Nonfinite final centroids on device")

    metadata = {
        "algorithm": "fixed_minibatch_cuda",
        "config": asdict(config),
        "final_exact_sse": final_sse,
        "steps": step,
        "termination_reason": "tolerance" if converged else "max_steps",
        "convergence_satisfied": converged,
        "stopping_rule": "3 successive checkpoints: movement <= tolerance and unchanged probe labels",
        "stable_checks_required": _STABLE_CHECKS,
        "stable_checks": stable_checks,
        "probe_size": probe_size,
        "probe_checks": probe_checks,
        "probe_indices": probe_indices.tolist(),
        "probe_role": "overlapping training-population stopping diagnostic",
        "sampling_policy": "uniform without replacement within batch; independent draws across steps; full batch uses input order",
        "sampling_rng": "numpy.default_rng(seed)",
        "probe_rng": "numpy.default_rng(SeedSequence([seed,1]))",
        "cluster_observation_counts": host_counts.tolist(),
        "observation_counts_include_C0": False,
        "points_processed": step * batch_size,
        "assignment_points_processed": step * batch_size + (probe_checks + 1) * probe_size + n,
        "input_dtype": "float32",
        "centroid_dtype": "float32",
        "accumulation_dtype": "float64",
        "empty_cluster_policy": "retain",
        "tie_policy": "lowest_index",
        "initialization": "provided_C0",
        "gpu": str(device.name),
        "compute_capability": tuple(device.compute_capability),
        "cuda_driver_version": driver.get_version(),
        "cuda_toolkit_nvrtc_version": runtime.get_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "numba": version("numba"),
        "numba-cuda": version("numba-cuda"),
        "threads_per_block": _THREADS,
        "max_blocks": _MAX_BLOCKS,
        "estimated_device_buffer_bytes": estimated_bytes,
        "device_total_bytes": int(total_bytes),
        "history_objective_policy": "batch/probe objectives only; final full SSE exclusively in final_exact_sse",
        "timing_scope": (
            "wall includes validation, context setup, allocations, transfers, compute and metadata; "
            "compute includes sampling, batch assignments, accumulation/updates, monitoring and final assignment; "
            "kernel_s and final_assignment_s overlap compute_s; "
            "upload_s is within setup_s; transfer_s includes upload, scalar downloads and result downloads"
        ),
    }

    timings = dict(
        compute_s=compute_time,
        sampling_s=sampling_time,
        batch_assignment_s=assignment_time,
        accumulation_update_s=update_time,
        monitoring_s=monitoring_time,
        final_assignment_s=final_assignment_time,
        kernel_s=kernel_time,
        setup_s=setup_time,
        upload_s=upload_time,
        scalar_transfer_s=scalar_transfer_time,
        download_s=download_time,
        transfer_s=upload_time + scalar_transfer_time + download_time,
        wall_s=perf_counter() - wall_start,
    )

    return RunResult(host_centroids, host_labels, history, timings, metadata)
