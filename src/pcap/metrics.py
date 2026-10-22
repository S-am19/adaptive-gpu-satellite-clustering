"""Evaluation metrics for satellite image clustering quality.

Computes exact and sampled quality metrics without memory blowup:
- Sum of Squared Errors (SSE) and Root Mean Squared Error (RMSE)
- Davies-Bouldin Index on reproducible point samples
- Silhouette Score on reproducible point samples
- ANN assignment recall against exact nearest-centroid labels
"""

import math
from numbers import Integral, Real

import numpy as np

from .contracts import RunResult, validate_features, validate_integer


def compute_sse(points: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact full-data SSE with float64 accumulation."""
    validate_features("points", points)
    validate_features("centroids", centroids)
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int32:
        raise TypeError("labels must be an int32 NumPy array")
    if labels.shape != (points.shape[0],):
        raise ValueError("labels must have shape [N]")

    delta = points.astype(np.float64) - centroids[labels].astype(np.float64)
    return float(np.sum(delta * delta, dtype=np.float64))


def compute_rmse(sse: float, n: int, d: int) -> float:
    """Compute Root Mean Squared Error: sqrt(SSE / (N * D))."""
    if not isinstance(sse, Real) or not math.isfinite(sse) or sse < 0:
        raise ValueError("sse must be finite and nonnegative")
    validate_integer("n", n)
    validate_integer("d", d)
    return math.sqrt(float(sse) / (int(n) * int(d)))


def compute_normalized_sse(sse: float, n: int) -> float:
    """Compute per-point SSE: SSE / N."""
    if not isinstance(sse, Real) or not math.isfinite(sse) or sse < 0:
        raise ValueError("sse must be finite and nonnegative")
    validate_integer("n", n)
    return float(sse) / int(n)


def compute_ann_recall(ann_labels: np.ndarray, exact_labels: np.ndarray) -> float:
    """Compute fraction of exact nearest-centroid label matches (ANN recall)."""
    if not isinstance(ann_labels, np.ndarray) or not isinstance(exact_labels, np.ndarray):
        raise TypeError("Labels must be NumPy arrays")
    if ann_labels.shape != exact_labels.shape:
        raise ValueError("ann_labels and exact_labels must have the same shape")
    if len(ann_labels) == 0:
        return 1.0
    return float(np.mean(ann_labels == exact_labels))


def _sample_for_metrics(points: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    n = points.shape[0]
    if n <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=max_points, replace=False))
    return np.ascontiguousarray(points[indices]), np.ascontiguousarray(labels[indices])


def compute_davies_bouldin(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 5000,
    seed: int = 0,
) -> float:
    """Compute Davies-Bouldin Index on a fixed reproducible sample of points.

    Lower values indicate better clustering (higher intra-cluster compactness
    relative to inter-cluster separation).
    """
    validate_features("points", points)
    validate_integer("max_points", max_points)
    sub_points, sub_labels = _sample_for_metrics(points, labels, max_points, seed)

    unique_labels = np.unique(sub_labels)
    k = len(unique_labels)
    if k <= 1:
        return 0.0

    # Cluster centroids and intra-cluster dispersions
    centroids = np.empty((k, sub_points.shape[1]), dtype=np.float64)
    dispersions = np.empty(k, dtype=np.float64)

    for idx, label in enumerate(unique_labels):
        cluster_pts = sub_points[sub_labels == label].astype(np.float64)
        c = cluster_pts.mean(axis=0)
        centroids[idx] = c
        dispersions[idx] = float(np.mean(np.sqrt(np.sum((cluster_pts - c) ** 2, axis=1))))

    # Inter-cluster centroid distances
    centroid_deltas = centroids[:, None, :] - centroids[None, :, :]
    centroid_dists = np.sqrt(np.sum(centroid_deltas ** 2, axis=2))

    # Davies-Bouldin R_ij = (s_i + s_j) / d(c_i, c_j)
    d_vals = []
    for i in range(k):
        r_i = []
        for j in range(k):
            if i != j:
                dist = centroid_dists[i, j]
                ratio = (dispersions[i] + dispersions[j]) / dist if dist > 1e-12 else 0.0
                r_i.append(ratio)
        d_vals.append(max(r_i) if r_i else 0.0)

    return float(np.mean(d_vals))


def compute_silhouette_score(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 3000,
    seed: int = 0,
) -> float:
    """Compute mean Silhouette Coefficient on a reproducible sample of points.

    Evaluates cluster cohesion vs separation in range [-1, 1]. Higher is better.
    """
    validate_features("points", points)
    validate_integer("max_points", max_points)
    sub_points, sub_labels = _sample_for_metrics(points, labels, max_points, seed)

    unique_labels, cluster_counts = np.unique(sub_labels, return_counts=True)
    k = len(unique_labels)
    if k <= 1 or len(sub_points) <= k:
        return 0.0

    # Pairwise Euclidean distances
    diff = sub_points[:, None, :].astype(np.float64) - sub_points[None, :, :].astype(np.float64)
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    silhouettes = np.empty(len(sub_points), dtype=np.float64)

    for i in range(len(sub_points)):
        own_label = sub_labels[i]
        own_mask = (sub_labels == own_label)

        # Intra-cluster mean distance a(i)
        own_count = np.sum(own_mask)
        if own_count > 1:
            a_i = np.sum(dist_matrix[i, own_mask]) / (own_count - 1)
        else:
            a_i = 0.0

        # Nearest cluster mean distance b(i)
        b_candidates = []
        for other_label in unique_labels:
            if other_label != own_label:
                other_mask = (sub_labels == other_label)
                if np.any(other_mask):
                    b_candidates.append(np.mean(dist_matrix[i, other_mask]))

        b_i = min(b_candidates) if b_candidates else 0.0

        max_ab = max(a_i, b_i)
        if max_ab > 1e-12 and own_count > 1:
            silhouettes[i] = (b_i - a_i) / max_ab
        else:
            silhouettes[i] = 0.0

    return float(np.mean(silhouettes))


def evaluate_clustering_quality(
    points: np.ndarray,
    result: RunResult,
    *,
    max_sample_points: int = 3000,
    seed: int = 0,
    ann_labels: np.ndarray | None = None,
) -> dict:
    """Comprehensive clustering evaluation dictionary for RunResult.

    Computes SSE, normalized SSE, RMSE, DBI, Silhouette, and optional ANN recall.
    """
    validate_features("points", points)
    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult instance")

    n, d = points.shape
    sse = result.sse
    rmse = compute_rmse(sse, n, d)
    norm_sse = compute_normalized_sse(sse, n)

    dbi = compute_davies_bouldin(
        points, result.labels, max_points=max_sample_points, seed=seed
    )
    sil = compute_silhouette_score(
        points, result.labels, max_points=max_sample_points, seed=seed
    )

    metrics = {
        "final_exact_sse": sse,
        "sse_per_point": norm_sse,
        "rmse": rmse,
        "davies_bouldin_index": dbi,
        "silhouette_score": sil,
        "evaluation_sample_size": min(n, max_sample_points),
        "total_points": n,
        "feature_dim": d,
        "cluster_k": len(result.centroids),
        "steps": result.metadata.get("steps", len(result.history)),
        "termination_reason": result.metadata.get("termination_reason", "unknown"),
        "wall_time_s": result.timings.get("wall_s", 0.0),
        "compute_time_s": result.timings.get("compute_s", 0.0),
    }

    if ann_labels is not None:
        metrics["ann_recall"] = compute_ann_recall(ann_labels, result.labels)

    return metrics
