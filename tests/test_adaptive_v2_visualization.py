import json

import numpy as np
import pytest

from pcap.adaptive_v2_visualization import (
    assign_exact_chunked,
    make_result_metadata,
    verify_sample_reconstruction,
)


def test_saved_sample_reconstruction_requires_exact_feature_and_index_match():
    full_X = np.arange(40, dtype=np.float32).reshape(10, 4)
    full_indices = np.array([100, 101, 102, 104, 106, 107, 108, 109, 110, 111], dtype=np.int64)
    picked = np.array([101, 104, 109], dtype=np.int64)
    sample_X = full_X[[1, 3, 7]].copy()
    np.testing.assert_array_equal(
        verify_sample_reconstruction(full_X, full_indices, sample_X, picked), [1, 3, 7]
    )
    with pytest.raises(ValueError, match='features differ'):
        verify_sample_reconstruction(full_X, full_indices, sample_X + 1, picked)
    with pytest.raises(ValueError, match='not a subset'):
        verify_sample_reconstruction(full_X, full_indices, sample_X,
                                     np.array([101, 105, 109]))


def test_chunked_exact_assignment_matches_direct_assignment_and_sse():
    points = np.array([[0, 0], [2, 0], [10, 10], [10, 11], [5, 5]], dtype=np.float32)
    centroids = np.array([[0, 0], [10, 10]], dtype=np.float32)
    labels, sse = assign_exact_chunked(points, centroids, chunk_size=2)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 0])
    assert sse == pytest.approx(55.0)
    with pytest.raises(ValueError):
        assign_exact_chunked(points, centroids, chunk_size=0)


def test_result_metadata_separates_sample_fit_sse_from_map_assignment():
    row = {
        'final_exact_sse': '12.5', 'metadata_summary': json.dumps({
            'algorithm': 'adaptive_v2_decayed_cuda', 'observation_budget_reached': True}),
        'X_sha256': 'samplehash', 'N': '100', 'K': '16', 'seed': '0',
        'repeat': '0', 'termination_reason': 'observation_budget',
        'convergence_satisfied': 'False', 'observations_processed': '1000', 'updates': '4',
    }
    result = make_result_metadata(
        method='Adaptive V2', row=row, history=[], full_X_hash='fullhash',
        map_points=120, reconstructed_fit_sse=12.5, centroid_path='centroids.npy',
    )
    assert result['sampled_fit_final_exact_sse'] == 12.5
    assert result['full_crop_map_assignment_sse'] is None
    assert result['fit_points'] == 100 and result['map_points'] == 120
    assert result['convergence_satisfied'] is False
    with pytest.raises(ValueError, match='do not reproduce'):
        make_result_metadata(
            method='Adaptive V2', row=row, history=[], full_X_hash='fullhash',
            map_points=120, reconstructed_fit_sse=13, centroid_path='centroids.npy',
        )
