"""Reconstruct full-crop labels from saved confirmation-study centroids.

This module does not fit a clustering model. It verifies the archived sampled
fit input, then assigns every valid pixel in the crop to the saved centroids.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .assignments_exact import assign_exact
from .preprocessing import Standardization, load_sentinel2


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_sample_reconstruction(full_X, full_pixel_indices, sample_X, sample_indices):
    """Require saved sample rows to exactly match reconstructed full-scene rows."""
    full_X = np.asarray(full_X)
    sample_X = np.asarray(sample_X)
    full_pixel_indices = np.asarray(full_pixel_indices)
    sample_indices = np.asarray(sample_indices)
    if (full_X.ndim != 2 or sample_X.ndim != 2 or full_X.shape[1] != 4
            or sample_X.shape[1] != 4 or full_X.dtype != np.float32
            or sample_X.dtype != np.float32 or not full_X.flags.c_contiguous
            or not sample_X.flags.c_contiguous):
        raise ValueError('Expected contiguous float32 four-band feature matrices')
    if (full_pixel_indices.shape != (len(full_X),)
            or sample_indices.ndim != 1
            or not np.issubdtype(full_pixel_indices.dtype, np.integer)
            or not np.issubdtype(sample_indices.dtype, np.integer)
            or np.any(np.diff(full_pixel_indices) <= 0)
            or np.any(np.diff(sample_indices) <= 0)):
        raise ValueError('Pixel indices must be strictly increasing integer vectors')
    positions = np.searchsorted(full_pixel_indices, sample_indices)
    if np.any(positions >= len(full_pixel_indices)):
        raise ValueError('Saved sample includes pixels absent from full reconstruction')
    if not np.array_equal(full_pixel_indices[positions], sample_indices):
        raise ValueError('Saved sample indices are not a subset of full valid pixels')
    if not np.array_equal(full_X[positions], sample_X):
        raise ValueError('Reconstructed features differ from the archived fit sample')
    return positions


def assign_exact_chunked(X, centroids, *, chunk_size=262144):
    """Exact lowest-index assignment with bounded temporary distance storage."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    labels = np.empty(len(X), dtype=np.int32)
    sse = 0.0
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        result = assign_exact(X[start:stop], centroids)
        labels[start:stop] = result.labels
        sse += float(np.sum(result.squared_distances, dtype=np.float64))
    return labels, sse


def make_result_metadata(*, method, row, history, full_X_hash, map_points,
                         reconstructed_fit_sse, centroid_path):
    """Keep archived sampled-fit SSE distinct from full-crop map assignment."""
    fit_sse = float(row['final_exact_sse'])
    if not np.isclose(reconstructed_fit_sse, fit_sse, rtol=1e-10, atol=1e-5):
        raise ValueError('Saved centroids do not reproduce the archived fit SSE')
    return {
        'algorithm': str(row['metadata_summary'] and json.loads(row['metadata_summary']).get('algorithm', method)),
        'method_label': method,
        'X_sha256': full_X_hash,
        'fit_input_X_sha256': row['X_sha256'],
        'fit_points': int(row['N']),
        'map_points': int(map_points),
        'K': int(row['K']),
        'seed': int(row['seed']),
        'repeat': int(row['repeat']),
        'fit_scope': 'Centroids fit on the archived uniformly sampled feature matrix; no refit performed.',
        'map_assignment': 'Exact nearest-centroid assignment over all valid pixels in the reconstructed crop.',
        'sampled_fit_final_exact_sse': fit_sse,
        'sampled_fit_sse_recomputed_from_saved_centroids': float(reconstructed_fit_sse),
        'full_crop_map_assignment_sse': None,
        'full_crop_map_assignment_sse_note': 'Not reported as a fit result; renderer only creates pixel labels.',
        'termination_reason': row['termination_reason'],
        'convergence_satisfied': row['convergence_satisfied'].lower() == 'true',
        'observation_budget_reached': (json.loads(row['metadata_summary']).get('observation_budget_reached')
                                       if row['metadata_summary'] else None),
        'observations_processed': int(row['observations_processed']) if row['observations_processed'] else None,
        'updates': int(row['updates']) if row['updates'] else None,
        'centroids_source': str(centroid_path),
        'history': history,
    }


def reconstruct_full_scene(study_dir):
    """Read the original SAFE window and apply frozen development scaling."""
    study_dir = Path(study_dir)
    prep_dir = study_dir / 'preparation'
    archived = json.loads((prep_dir / 'metadata.json').read_text())
    scene_meta = archived['scene']
    band_paths = scene_meta['band_paths']
    scl_path = scene_meta['scl_path']
    product_metadata_path = archived['product_metadata_path']
    missing = [p for p in [*band_paths.values(), scl_path, product_metadata_path]
               if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('Required original Sentinel-2 inputs are missing: ' + ', '.join(missing))
    standardization = Standardization.from_json(json.dumps(scene_meta['standardization']))
    crop = tuple(scene_meta['window'])
    pixel_count = int(scene_meta['crop_grid']['height'] * scene_meta['crop_grid']['width'])
    prepared = load_sentinel2(band_paths, scl_path, product_metadata_path,
                              window=crop, max_crop_pixels=pixel_count,
                              sample_size=None, seed=0)
    full_scene = prepared.standardized(standardization)
    if len(full_scene.X) != scene_meta['valid_pixels']:
        raise ValueError('Reconstructed valid-pixel count differs from archived metadata')
    if full_scene.valid_mask.shape != (scene_meta['crop_grid']['height'], scene_meta['crop_grid']['width']):
        raise ValueError('Reconstructed crop shape differs from archived metadata')
    if (len(full_scene.X) != int(full_scene.valid_mask.sum())
            or not np.array_equal(np.flatnonzero(full_scene.valid_mask), full_scene.pixel_indices)):
        raise ValueError('Full-scene features do not map one-to-one onto every valid crop pixel')
    if list(full_scene.metadata['crop_grid']['transform']) != scene_meta['crop_grid']['transform']:
        raise ValueError('Reconstructed geospatial transform differs from archived metadata')
    if full_scene.metadata['crop_grid']['crs'] != scene_meta['crop_grid']['crs']:
        raise ValueError('Reconstructed CRS differs from archived metadata')

    sample_X = np.load(prep_dir / 'X.npy', allow_pickle=False)
    sample_indices = np.load(prep_dir / 'pixel_indices.npy', allow_pickle=False)
    archived_mask = np.load(prep_dir / 'valid_mask.npy', allow_pickle=False)
    if sha256_array(sample_X) != scene_meta['X_sha256']:
        raise ValueError('Archived sampled feature matrix checksum mismatch')
    if sha256_array(sample_indices) != scene_meta['pixel_indices_sha256']:
        raise ValueError('Archived sampled pixel-index checksum mismatch')
    if not np.array_equal(archived_mask, full_scene.valid_mask):
        raise ValueError('Reconstructed valid-pixel mask differs from archived mask')
    verify_sample_reconstruction(full_scene.X, full_scene.pixel_indices, sample_X, sample_indices)
    return full_scene
