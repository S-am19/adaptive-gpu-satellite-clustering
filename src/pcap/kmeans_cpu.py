"""Full Lloyd K-Means: compiled sequential loops, no parallel/fastmath mode.

Inputs/centroids are float32; distances, sums, movement and SSE use float64.
Each history SSE is an exact assignment objective at that step's centroids.
Timing includes JIT compilation on a cold call and is never labeled warmed.
"""

from dataclasses import asdict
from importlib.metadata import version
import platform
from time import perf_counter

import numpy as np
from numba import njit

from .assignments_exact import _accumulate, _assign
from .contracts import FitConfig, RunResult, validate_points_centroids


@njit
def _lloyd(points, initial, max_steps, tolerance):
    centroids = initial.copy()
    objectives = np.empty(max_steps + 1, dtype=np.float64)
    movements = np.zeros(max_steps + 1, dtype=np.float64)
    labels, distances = _assign(points, centroids)
    objectives[0] = distances.sum()
    converged = False
    steps = 0
    for step in range(1, max_steps + 1):
        sums, counts = _accumulate(points, labels, centroids.shape[0])
        updated = centroids.copy()
        for j in range(centroids.shape[0]):
            if counts[j] > 0:
                for d in range(centroids.shape[1]):
                    updated[j, d] = sums[j, d] / counts[j]
        max_squared_shift = 0.0
        for j in range(centroids.shape[0]):
            shift = 0.0
            for d in range(centroids.shape[1]):
                delta = np.float64(updated[j, d]) - np.float64(centroids[j, d])
                shift += delta * delta
            max_squared_shift = max(max_squared_shift, shift)
        centroids = updated
        movements[step] = np.sqrt(max_squared_shift)
        steps = step
        if movements[step] <= tolerance:
            converged = True
            break
        if step < max_steps:
            labels, distances = _assign(points, centroids)
            objectives[step] = distances.sum()
    # The final assignment is performed once by the separately timed wrapper.
    # objectives contains the initial objective and intermediate steps only.
    return centroids, objectives[:steps], movements[:steps + 1], converged


@njit
def _final_exact_assignment(points, centroids):
    labels, distances = _assign(points, centroids)
    return labels, distances.sum()


def fit_cpu(X, C0, config) -> RunResult:
    start = perf_counter()
    if not isinstance(config, FitConfig):
        raise TypeError("config must be FitConfig")
    validate_points_centroids(X, C0)
    if C0.shape[0] != config.k or config.k > X.shape[0]:
        raise ValueError("C0 rows must equal config.k, with k <= N")
    if config.batch_size is not None or config.ann_effort is not None:
        raise ValueError("full exact CPU fitting requires batch_size=ann_effort=None")
    compute_start = perf_counter()
    centroids, objectives, movements, converged = _lloyd(
        X, C0, config.max_steps, config.tolerance,
    )
    final_assignment_start = perf_counter()
    labels, final_sse = _final_exact_assignment(X, centroids)
    compute_end = perf_counter()
    final_assignment_time = compute_end - final_assignment_start
    compute_time = compute_end - compute_start
    objectives = np.append(objectives, final_sse)
    history = [
        {"step": step, "points_processed": step * X.shape[0],
         "assignment_points_processed": (step + 1) * X.shape[0],
         "centroid_displacement": float(movements[step]), "sse": float(objective)}
        for step, objective in enumerate(objectives)
    ]
    return RunResult(
        centroids=centroids, labels=labels, history=history,
        timings={"compute_s": compute_time, "wall_s": perf_counter() - start,
                 "final_assignment_s": final_assignment_time},
        metadata={
            "algorithm": "sequential_lloyd", "config": asdict(config),
            "final_exact_sse": float(final_sse),
            "termination_reason": "tolerance" if converged else "max_steps",
            "steps": len(history) - 1, "input_dtype": "float32",
            "accumulation_dtype": "float64", "empty_cluster_policy": "retain",
            "tie_policy": "lowest_index", "initialization": "provided_C0",
            "timing_scope": (
                "validation+fit+history; compute includes initial/final exact assignment; "
                "final_assignment_s includes SSE reduction and is a subset of compute_s/wall_s"
            ),
            "jit_policy": "cold calls include compilation; warm explicitly before benchmarking",
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "numpy": version("numpy"), "numba": version("numba"),
        },
    )
