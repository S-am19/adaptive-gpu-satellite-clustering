"""Unit tests for clustering evaluation metrics."""

import math
import numpy as np
import pytest

from pcap.contracts import RunResult
from pcap.metrics import (
    compute_ann_recall,
    compute_davies_bouldin,
    compute_normalized_sse,
    compute_rmse,
    compute_silhouette_score,
    compute_sse,
    evaluate_clustering_quality,
)


def test_sse_and_rmse_exact():
    points = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]], dtype=np.float32)
    centroids = np.array([[0.0, 1.0], [10.0, 11.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    # Distances squared: (0-1)^2 = 1, (2-1)^2 = 1, (10-11)^2 = 1, (12-11)^2 = 1 => SSE = 4.0
    sse = compute_sse(points, centroids, labels)
    assert sse == pytest.approx(4.0)

    # RMSE: sqrt(4.0 / (4 * 2)) = sqrt(0.5)
    rmse = compute_rmse(sse, 4, 2)
    assert rmse == pytest.approx(math.sqrt(0.5))

    # Normalized SSE: 4.0 / 4 = 1.0
    norm_sse = compute_normalized_sse(sse, 4)
    assert norm_sse == pytest.approx(1.0)


def test_ann_recall_calculation():
    exact = np.array([0, 1, 2, 3], dtype=np.int32)
    ann_perfect = np.array([0, 1, 2, 3], dtype=np.int32)
    ann_half = np.array([0, 1, 0, 0], dtype=np.int32)

    assert compute_ann_recall(ann_perfect, exact) == 1.0
    assert compute_ann_recall(ann_half, exact) == 0.5


def test_davies_bouldin_separated_clusters():
    # Two very tightly clustered groups far apart
    g1 = np.array([[-100.0, -100.0], [-100.1, -100.1], [-99.9, -99.9]], dtype=np.float32)
    g2 = np.array([[100.0, 100.0], [100.1, 100.1], [99.9, 99.9]], dtype=np.float32)
    points = np.vstack([g1, g2])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    dbi = compute_davies_bouldin(points, labels, max_points=10)
    # Compact intra-dispersion (~0.1) vs inter-centroid distance (~282) => DBI is very close to 0
    assert 0.0 <= dbi < 0.01


def test_silhouette_score_separated_clusters():
    # Two very tightly clustered groups far apart
    g1 = np.array([[-50.0, 0.0], [-50.1, 0.0], [-49.9, 0.0]], dtype=np.float32)
    g2 = np.array([[50.0, 0.0], [50.1, 0.0], [49.9, 0.0]], dtype=np.float32)
    points = np.vstack([g1, g2])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    sil = compute_silhouette_score(points, labels, max_points=10)
    # Nearly perfect cluster separation -> Silhouette score > 0.95
    assert sil > 0.95


def test_evaluate_clustering_quality_wrapper():
    points = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]], dtype=np.float32)
    centroids = np.array([[0.0, 1.0], [10.0, 11.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    result = RunResult(
        centroids=centroids,
        labels=labels,
        history=[],
        timings={"wall_s": 0.05, "compute_s": 0.04},
        metadata={"final_exact_sse": 4.0, "steps": 5, "termination_reason": "tolerance"},
    )

    ann_labels = np.array([0, 0, 1, 0], dtype=np.int32)  # 3/4 matches
    report = evaluate_clustering_quality(points, result, max_sample_points=10, ann_labels=ann_labels)

    assert report["final_exact_sse"] == 4.0
    assert report["rmse"] == pytest.approx(math.sqrt(0.5))
    assert report["ann_recall"] == 0.75
    assert report["total_points"] == 4
    assert report["cluster_k"] == 2
    assert report["termination_reason"] == "tolerance"
