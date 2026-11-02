"""Render archived Adaptive V2 and exact-CUDA centroids on a full SAFE crop.

No clustering fit is run. The saved sampled fit is verified against the original
SAFE, then its centroids are assigned to every valid pixel in the crop.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pcap.adaptive_v2_visualization import (  # noqa: E402
    assign_exact_chunked, make_result_metadata, reconstruct_full_scene, sha256_array,
)


def _find_run(study_dir, method, seed, repeat):
    path = Path(study_dir) / 'confirmatory' / 'individual_runs.csv'
    with path.open(newline='') as handle:
        matches = [r for r in csv.DictReader(handle)
                   if r['method'] == method and int(r['seed']) == seed
                   and int(r['repeat']) == repeat]
    if len(matches) != 1:
        raise ValueError(f'Expected one run for {method}, seed={seed}, repeat={repeat}; found {len(matches)}')
    return matches[0]


def _history(study_dir, method, seed, repeat):
    path = Path(study_dir) / 'confirmatory' / 'batch_trajectories.csv'
    with path.open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['method'] == method and int(r['seed']) == seed
                and int(r['repeat']) == repeat]
    result = []
    for row in rows:
        result.append({
            'step': int(row['step']),
            'batch_size': int(row['batch_size']),
            'ann_effort': None,
            'points_processed': int(row['points_processed']),
            'centroid_displacement': float(row['centroid_displacement']),
            'forgetting_factor': float(row['forgetting_factor']) if row['forgetting_factor'] else None,
            'controller_decision': row['controller_decision'] or None,
            'adaptation_frozen': row['adaptation_frozen'].lower() == 'true'
                                 if row['adaptation_frozen'] else None,
        })
    return result


def _write_scene(scene, scene_dir, source_metadata):
    scene_dir.mkdir(parents=True)
    np.save(scene_dir / 'X.npy', scene.X, allow_pickle=False)
    np.save(scene_dir / 'valid_mask.npy', scene.valid_mask, allow_pickle=False)
    np.save(scene_dir / 'pixel_indices.npy', scene.pixel_indices, allow_pickle=False)
    metadata = dict(scene.metadata)
    metadata['standardization'] = source_metadata['scene']['standardization']
    metadata['X_sha256'] = sha256_array(scene.X)
    metadata['crop_grid'] = {
        'height': int(scene.valid_mask.shape[0]),
        'width': int(scene.valid_mask.shape[1]),
        'transform': list(metadata['crop_grid']['transform']),
        'crs': metadata['crop_grid']['crs'],
    }
    metadata['visualization_scope'] = 'Full valid-pixel crop reconstructed from original Level-2A SAFE.'
    (scene_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata


def _write_model(study_dir, output_dir, method, display_name, seed, repeat, scene):
    row = _find_run(study_dir, method, seed, repeat)
    study_dir = Path(study_dir)
    centroid_path = study_dir / row['final_centroids_path']
    if not centroid_path.is_file():
        raise FileNotFoundError(f'Saved final centroids missing: {centroid_path}')
    centroids = np.load(centroid_path, allow_pickle=False)
    if centroids.shape != (int(row['K']), 4) or centroids.dtype != np.float32:
        raise ValueError(f'Unexpected centroid shape or dtype in {centroid_path}')
    fit_X = np.load(study_dir / 'preparation' / 'X.npy', allow_pickle=False)
    if sha256_array(fit_X) != row['X_sha256']:
        raise ValueError('Run input hash does not match archived preparation matrix')
    _, fit_sse = assign_exact_chunked(fit_X, centroids)
    if not np.isclose(fit_sse, float(row['final_exact_sse']), rtol=1e-10, atol=1e-5):
        raise ValueError('Saved centroids fail archived sampled-fit SSE verification')
    labels, map_sse = assign_exact_chunked(scene.X, centroids)
    model_dir = output_dir / f'model_{method}_seed{seed}_rep{repeat}'
    model_dir.mkdir(parents=True)
    np.save(model_dir / 'labels.npy', labels, allow_pickle=False)
    np.save(model_dir / 'centroids.npy', centroids, allow_pickle=False)
    result = make_result_metadata(
        method=display_name, row=row,
        history=_history(study_dir, method, seed, repeat),
        full_X_hash=sha256_array(scene.X), map_points=len(scene.X),
        reconstructed_fit_sse=fit_sse, centroid_path=str(centroid_path),
    )
    result['map_title'] = f'{display_name} — exact labels on full valid crop'
    result['full_crop_assignment_sse_diagnostic_only'] = map_sse
    result['label_dtype'] = str(labels.dtype)
    result['label_range'] = [int(labels.min()), int(labels.max())]
    (model_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return model_dir, result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render_confirmation(study_dir, output_dir, *, seed=0, repeat=0):
    from render_scene import render
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import Affine

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Output path already exists; refusing to overwrite artifacts')
    study_dir = Path(study_dir)
    archived_meta = json.loads((study_dir / 'preparation' / 'metadata.json').read_text())
    scene = reconstruct_full_scene(study_dir)
    output_dir.mkdir(parents=True)
    scene_dir = output_dir / 'scene_full_valid'
    scene_meta = _write_scene(scene, scene_dir, archived_meta)

    method_specs = [
        ('ADAPTIVE_V2', 'Adaptive V2 (decayed-count CUDA Mini-Batch)'),
        ('G0_CUDA', 'Exact CUDA G0'),
    ]
    render_dirs = {}
    model_summaries = {}
    for method, display_name in method_specs:
        model_dir, result = _write_model(study_dir, output_dir, method, display_name,
                                         seed, repeat, scene)
        render_dir = output_dir / f'render_{method}_seed{seed}_rep{repeat}'
        render(scene_dir, model_dir, render_dir)
        render_dirs[method] = render_dir
        model_summaries[method] = result

        tif = render_dir / 'clusters.tif'
        with rasterio.open(tif) as dataset:
            expected_transform = Affine(*scene_meta['crop_grid']['transform'])
            if ((dataset.height, dataset.width) != scene.valid_mask.shape
                    or dataset.count != 1 or dataset.crs.to_string() != scene_meta['crop_grid']['crs']
                    or dataset.transform != expected_transform or dataset.nodata != -1):
                raise ValueError(f'GeoTIFF georeferencing validation failed: {tif}')
            grid = dataset.read(1)
            if not np.all(grid[~scene.valid_mask] == -1):
                raise ValueError(f'Masked pixels are not nodata in {tif}')
            if int(np.count_nonzero(grid >= 0)) != int(scene.valid_mask.sum()):
                raise ValueError(f'Valid-pixel label count mismatch in {tif}')
            if np.any((grid[scene.valid_mask] < 0) | (grid[scene.valid_mask] >= 16)):
                raise ValueError(f'Cluster IDs outside 0..15 in {tif}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, method in zip(axes, ('ADAPTIVE_V2', 'G0_CUDA')):
        image = plt.imread(render_dirs[method] / 'clusters.png')
        ax.imshow(image)
        ax.axis('off')
    comparison = output_dir / 'adaptive_v2_vs_exact_cuda.png'
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    summary = {
        'source_study': str(study_dir),
        'product_id': scene_meta['calibration']['product_id'],
        'acquisition_date': '2024-06-28',
        'seed': seed,
        'repeat': repeat,
        'crop_shape': list(scene.valid_mask.shape),
        'valid_pixels': int(scene.valid_mask.sum()),
        'invalid_pixels': int((~scene.valid_mask).sum()),
        'crs': scene_meta['crop_grid']['crs'],
        'transform': scene_meta['crop_grid']['transform'],
        'models': model_summaries,
        'source_hashes': {
            'confirmation_manifest_sha256': _sha256(study_dir / 'confirmatory' / 'manifest.json'),
            'individual_runs_sha256': _sha256(study_dir / 'confirmatory' / 'individual_runs.csv'),
            'batch_trajectories_sha256': _sha256(study_dir / 'confirmatory' / 'batch_trajectories.csv'),
            'preparation_metadata_sha256': _sha256(study_dir / 'preparation' / 'metadata.json'),
            'renderer_sha256': _sha256(Path(__file__).with_name('render_scene.py')),
        },
        'comparison_figure': comparison.name,
        'interpretation': ('Labels are exact nearest-centroid assignments on every valid crop pixel. '
                           'Centroids are loaded from the archived fit; no model was refit. '
                           'Archived fit SSE remains the sampled training-set SSE.'),
    }
    (output_dir / 'visualization_manifest.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    files = sorted(p for p in output_dir.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    (output_dir / 'checksums.sha256').write_text(''.join(
        f'{_sha256(path)}  {path.relative_to(output_dir)}\n' for path in files))
    for path in files:
        expected = next(line.split()[0] for line in (output_dir / 'checksums.sha256').read_text().splitlines()
                        if line.split(maxsplit=1)[1] == str(path.relative_to(output_dir)))
        if _sha256(path) != expected:
            raise ValueError(f'Output checksum verification failed: {path}')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path,
                        default=ROOT / 'outputs/20261101-independent-confirmation-study')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    print(render_confirmation(args.study_dir, args.output_dir, seed=args.seed, repeat=args.repeat))


if __name__ == '__main__':
    main()
