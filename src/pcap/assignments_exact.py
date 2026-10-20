"""Sequential exact assignment and reusable per-cluster statistics."""

from time import perf_counter

import numpy as np
from numba import njit

from .contracts import (
    AssignmentResult, ClusterStats, validate_features, validate_integer,
    validate_points_centroids,
)


@njit
def _assign(points, centroids):
    labels = np.empty(points.shape[0], dtype=np.int32)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for i in range(points.shape[0]):
        best_distance = np.inf
        best_label = 0
        for j in range(centroids.shape[0]):
            distance = 0.0
            for d in range(points.shape[1]):
                delta = np.float64(points[i, d]) - np.float64(centroids[j, d])
                distance += delta * delta
            # Strict comparison preserves the lowest index on exact ties.
            if distance < best_distance:
                best_distance = distance
                best_label = j
        labels[i] = best_label
        distances[i] = best_distance
    return labels, distances


@njit
def _accumulate(points, labels, k):
    sums = np.zeros((k, points.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(points.shape[0]):
        label = labels[i]
        counts[label] += 1
        for d in range(points.shape[1]):
            sums[label, d] += np.float64(points[i, d])
    return sums, counts


def assign_exact(points, centroids) -> AssignmentResult:
    """Return float64 squared distances; first call may include JIT time."""
    validate_points_centroids(points, centroids)
    start = perf_counter()
    labels, distances = _assign(points, centroids)
    return AssignmentResult(labels, distances, perf_counter() - start)


def accumulate_cluster_stats(points, labels, k) -> ClusterStats:
    validate_features("points", points)
    validate_integer("k", k)
    if k > np.iinfo(np.int32).max:
        raise ValueError("k exceeds int32 label capacity")
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int32:
        raise TypeError("labels must be a NumPy int32 array")
    if labels.shape != (points.shape[0],) or not labels.flags.c_contiguous:
        raise ValueError("labels must be contiguous with shape [N]")
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("labels must be in 0..k-1")
    sums, counts = _accumulate(points, labels, k)
    return ClusterStats(sums, counts)


class ExactCPUBackend:
    """Stateless adapter; no CUDA imports, index, or search-effort control."""

    def prepare(self, centroids, effort=None):
        validate_features("centroids", centroids)
        if effort is not None:
            raise ValueError("exact CPU assignment requires effort=None")

    def assign(self, points, centroids, effort=None):
        self.prepare(centroids, effort)
        return assign_exact(points, centroids)

    def refresh(self, new_centroids):
        self.prepare(new_centroids)
