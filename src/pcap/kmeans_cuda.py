"""Exact Lloyd K-Means with custom Numba-CUDA kernels.

Import this optional module only when CUDA is needed. Shared contracts and CPU
modules never import it. All point-wise work runs on the GPU; host loops schedule
Lloyd steps and collect scalar diagnostics. Input and working arrays stay on the
device throughout fitting. There is no CPU fallback or out-of-core streaming.
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


_THREADS = 256
_MAX_BLOCKS = 1024


def _blocks(size):
    return min(_MAX_BLOCKS, (size + _THREADS - 1) // _THREADS)


@cuda.jit
def _assign_kernel(points, centroids, labels, distances):
    # Grid-stride tiles bound launch size without changing full Lloyd semantics.
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
def _reset_stats_kernel(sums, counts):
    for i in range(cuda.grid(1), sums.size, cuda.gridsize(1)):
        sums[i // sums.shape[1], i % sums.shape[1]] = 0.0
    for j in range(cuda.grid(1), counts.size, cuda.gridsize(1)):
        counts[j] = 0


@cuda.jit
def _accumulate_kernel(points, labels, sums, counts):
    for i in range(cuda.grid(1), points.shape[0], cuda.gridsize(1)):
        label = labels[i]
        cuda.atomic.add(counts, label, 1)
        for d in range(points.shape[1]):
            cuda.atomic.add(sums, (label, d), float64(points[i, d]))


@cuda.jit
def _update_kernel(centroids, sums, counts, squared_shifts):
    # One owner per centroid: no update races, including empty centroids.
    for j in range(cuda.grid(1), centroids.shape[0], cuda.gridsize(1)):
        shift = 0.0
        if counts[j] > 0:
            for d in range(centroids.shape[1]):
                old_value = float64(centroids[j, d])
                centroids[j, d] = sums[j, d] / counts[j]
                # Compute movement after rounding the stored centroid to float32.
                delta = float64(centroids[j, d]) - old_value
                shift += delta * delta
        squared_shifts[j] = shift


@cuda.jit
def _reduce_kernel(values, partials, use_max):
    """Fixed-size block reduction, reused for SSE and squared movement."""
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


def _required_device_bytes(n, d, k):
    # X, C, labels, distances, sums, counts, shifts, reduction scratch, scalars.
    return (4 * n * d + 4 * k * d + 12 * n + 8 * k * d + 16 * k
            + 8 * (_blocks(n) + _blocks(k) + 2))


def fit_cuda(X, C0, config) -> RunResult:
    """Fit exact full-data Lloyd K-Means on the current visible CUDA device.

    Same strict host-array/configuration validation as fit_cpu. All returned
    arrays are host arrays. Timing intervals overlap intentionally: final pass,
    scalar transfers and kernel intervals are subsets of compute_s, not additions
    to it. Cold runs include JIT compilation; this is not a benchmark runner.
    """
    wall_start = perf_counter()
    if not isinstance(config, FitConfig):
        raise TypeError("config must be FitConfig")
    validate_points_centroids(X, C0)
    if C0.shape[0] != config.k or config.k > X.shape[0]:
        raise ValueError("C0 rows must equal config.k, with k <= N")
    if config.batch_size is not None or config.ann_effort is not None:
        raise ValueError("full exact CUDA fitting requires batch_size=ann_effort=None")
    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError("fit_cuda requires real hardware, not CUDA simulator mode")
    if not cuda.is_available():
        raise RuntimeError("CUDA unavailable; on WSL2 use scripts/run_cuda_tests.sh for driver discovery")
    device = cuda.get_current_device()
    if device.compute_capability < (6, 0):
        raise RuntimeError("float64 atomic accumulation requires compute capability >= 6.0")
    cuda.synchronize()
    n, d = X.shape
    k = config.k
    estimated_bytes = _required_device_bytes(n, d, k)
    free_bytes, total_bytes = cuda.current_context().get_memory_info()
    if estimated_bytes > 0.8 * free_bytes:
        raise MemoryError(
            f"Resident CUDA buffers require {estimated_bytes} bytes; "
            f"free device memory is {free_bytes}. Use a smaller input crop."
        )

    setup_start = perf_counter()
    upload_start = perf_counter()
    points = cuda.to_device(X)
    centroids = cuda.to_device(C0)
    cuda.synchronize()
    upload_time = perf_counter() - upload_start
    labels = cuda.device_array(n, dtype=np.int32)
    distances = cuda.device_array(n, dtype=np.float64)
    sums = cuda.device_array((k, d), dtype=np.float64)
    counts = cuda.device_array(k, dtype=np.int64)
    shifts = cuda.device_array(k, dtype=np.float64)
    sse_partials = cuda.device_array(_blocks(n), dtype=np.float64)
    shift_partials = cuda.device_array(_blocks(k), dtype=np.float64)
    sse_scalar = cuda.device_array(1, dtype=np.float64)
    shift_scalar = cuda.device_array(1, dtype=np.float64)
    cuda.synchronize()
    setup_time = perf_counter() - setup_start
    kernel_time = 0.0
    scalar_transfer_time = 0.0

    def read_scalar(buffer):
        nonlocal scalar_transfer_time
        start = perf_counter()
        value = float(buffer.copy_to_host()[0])
        cuda.synchronize()
        scalar_transfer_time += perf_counter() - start
        return value

    def assign_and_evaluate():
        _assign_kernel[_blocks(n), _THREADS](points, centroids, labels, distances)
        _reduce(distances, sse_partials, sse_scalar, False)

    def update():
        _reset_stats_kernel[_blocks(k * d), _THREADS](sums, counts)
        _accumulate_kernel[_blocks(n), _THREADS](points, labels, sums, counts)
        _update_kernel[_blocks(k), _THREADS](centroids, sums, counts, shifts)
        _reduce(shifts, shift_partials, shift_scalar, True)

    compute_start = perf_counter()
    _, elapsed = _timed_kernels(assign_and_evaluate)
    kernel_time += elapsed
    history = [{"step": 0, "points_processed": 0, "assignment_points_processed": n,
                "centroid_displacement": 0.0, "sse": read_scalar(sse_scalar)}]
    converged = False
    final_assignment_time = 0.0
    for step in range(1, config.max_steps + 1):
        _, elapsed = _timed_kernels(update)
        kernel_time += elapsed
        displacement = math.sqrt(read_scalar(shift_scalar))
        converged = displacement <= config.tolerance
        is_final = converged or step == config.max_steps
        # Every assignment uses the updated centroids. The last pass is executed
        # exactly once, also when exiting by max_steps, and is timed separately.
        final_start = perf_counter()
        _, elapsed = _timed_kernels(assign_and_evaluate)
        kernel_time += elapsed
        sse = read_scalar(sse_scalar)
        if is_final:
            final_assignment_time = perf_counter() - final_start
        history.append({"step": step, "points_processed": step * n,
                        "assignment_points_processed": (step + 1) * n,
                        "centroid_displacement": displacement, "sse": sse})
        if is_final:
            break
    cuda.synchronize()
    compute_time = perf_counter() - compute_start
    download_start = perf_counter()
    host_centroids = centroids.copy_to_host()
    host_labels = labels.copy_to_host()
    cuda.synchronize()
    download_time = perf_counter() - download_start
    metadata = {
        "algorithm": "cuda_exact_lloyd", "config": asdict(config),
        "final_exact_sse": sse, "steps": step,
        "termination_reason": "tolerance" if converged else "max_steps",
        "input_dtype": "float32", "accumulation_dtype": "float64",
        "empty_cluster_policy": "retain", "tie_policy": "lowest_index",
        "initialization": "provided_C0", "gpu": str(device.name),
        "compute_capability": tuple(device.compute_capability),
        "cuda_driver_version": driver.get_version(),
        "cuda_toolkit_nvrtc_version": runtime.get_version(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": version("numpy"), "numba": version("numba"),
        "numba-cuda": version("numba-cuda"),
        "threads_per_block": _THREADS, "max_blocks": _MAX_BLOCKS,
        "estimated_device_buffer_bytes": estimated_bytes,
        "device_total_bytes": int(total_bytes), "shape": (n, d),
        "memory_policy": "resident X and buffers; grid-stride tiles; no N*K distance matrix",
        "reduction_policy": "float64 atomic sums (order may vary), int64 counts, block-tree SSE/max",
        "jit_policy": "cold host and CUDA-event intervals may include compilation stalls; warm explicitly",
        "timing_scope": (
            "wall includes validation, context, allocation, transfers, compute and result metadata; "
            "compute includes scalar downloads, monitoring and final exact pass; "
            "final_assignment_s and kernel_s overlap compute_s; "
            "upload_s is within setup_s; transfer_s includes upload, scalar downloads and result download"
        ),
    }
    return RunResult(
        host_centroids, host_labels, history,
        {"compute_s": compute_time, "final_assignment_s": final_assignment_time,
         "kernel_s": kernel_time, "setup_s": setup_time, "upload_s": upload_time,
         "scalar_transfer_s": scalar_transfer_time, "download_s": download_time,
         "transfer_s": upload_time + scalar_transfer_time + download_time,
         "wall_s": perf_counter() - wall_start},
        metadata,
    )
