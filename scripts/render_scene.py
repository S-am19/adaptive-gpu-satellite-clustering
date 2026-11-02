"""Render a prepared four-band scene and a saved model, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments'))


def render(scene_dir, model_dir, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    from affine import Affine
    from plot_results import render_cluster_map
    scene_dir, model_dir, output_dir = map(Path, (scene_dir, model_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory; images are never overwritten')
    X = np.load(scene_dir/'X.npy', allow_pickle=False)
    mask = np.load(scene_dir/'valid_mask.npy', allow_pickle=False)
    indices = np.load(scene_dir/'pixel_indices.npy', allow_pickle=False)
    labels = np.load(model_dir/'labels.npy', allow_pickle=False)
    C = np.load(model_dir/'centroids.npy', allow_pickle=False)
    meta = json.loads((scene_dir/'metadata.json').read_text())
    result = json.loads((model_dir/'result.json').read_text())
    x_hash = hashlib.sha256(X.tobytes()).hexdigest()
    if x_hash != meta['X_sha256'] or x_hash != result['X_sha256']:
        raise ValueError('Scene/model feature hashes differ')
    from pcap.assignments_exact import assign_exact
    exact = assign_exact(X, C)
    if not np.array_equal(exact.labels, labels):
        raise ValueError('Model labels do not match exact final centroids')
    if X.shape[1] != 4 or len(labels) != len(X):
        raise ValueError('Four features and one label per point required')
    reflectance = X.astype(np.float64)
    if 'standardization' in meta:
        stats = meta['standardization']
        reflectance = reflectance*np.asarray(stats['scale'])+np.asarray(stats['mean'])
    output_dir.mkdir(parents=True, exist_ok=False)
    render_cluster_map(labels, mask, indices, output_dir/'clusters.png',
                       title=result.get('map_title', 'Satellite pixel cluster IDs'))
    grid = np.full(mask.shape, -1, np.int32)
    grid.ravel()[indices] = labels
    geometry = meta['crop_grid']
    with rasterio.open(output_dir/'clusters.tif', 'w', driver='GTiff', height=mask.shape[0],
                       width=mask.shape[1], count=1, dtype='int32', nodata=-1,
                       crs=geometry['crs'], transform=Affine(*geometry['transform']), compress='deflate') as dst:
        dst.write(grid,1)
    # Display stretches use this image only and never alter clustering features.
    for name, bands in [('true_color',(2,1,0)), ('false_color_nir',(3,2,1))]:
        values = reflectance[:,bands]
        lo,hi = np.percentile(values, [2,98], axis=0)
        pixels = np.clip((values-lo)/np.maximum(hi-lo,1e-12),0,1)
        rgb = np.zeros((*mask.shape,3),np.float32)
        rgb.reshape(-1,3)[indices] = pixels
        plt.imsave(output_dir/f'{name}.png',rgb)
    history = result['history']
    fig, axes = plt.subplots(2,1,sharex=True)
    axes[0].step([h['step'] for h in history], [h.get('batch_size',len(X)) for h in history],where='post')
    axes[0].set_ylabel('Batch observations')
    axes[1].step([h['step'] for h in history], [h.get('ann_effort') or 0 for h in history],where='post')
    axes[1].set_ylabel('ANN effort (0=none)'); axes[1].set_xlabel('Update step')
    fig.tight_layout(); fig.savefig(output_dir/'adaptation.png'); plt.close(fig)
    (output_dir/'provenance.json').write_text(json.dumps(dict(scene=meta, model=result,
        display='per-channel 2nd–98th percentile stretch; B04/B03/B02 and B08/B04/B03; black=masked/unsampled'),indent=2))
    artifacts = sorted(p for p in output_dir.iterdir() if p.is_file())
    (output_dir/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in artifacts))
    return output_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(render(args.scene_dir,args.model_dir,args.output_dir))


if __name__ == '__main__':
    main()
