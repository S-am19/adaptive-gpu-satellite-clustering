"""Sentinel-2 Level-2A batch extraction and standardization pipeline.

Processes authentic ESA Copernicus .SAFE granules or directories containing
band JP2 assets, applies SCL quality masking, fits non-leaking standardization
statistics on training scenes, and exports cropped feature arrays for clustering.
"""

import argparse
from glob import glob
import json
from pathlib import Path
import sys

import numpy as np

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcap.preprocessing import fit_standardization, load_sentinel2


def find_scene_assets(scene_dir: Path):
    """Find B02, B03, B04, B08, SCL and metadata XML in a SAFE directory."""
    xml_candidates = list(scene_dir.rglob("MTD_MSIL2A.xml")) + list(scene_dir.rglob("MTD_TL.xml"))
    metadata_path = str(xml_candidates[0]) if xml_candidates else None

    # Search for 10m bands
    band_paths = {}
    for band in ("B02", "B03", "B04", "B08"):
        candidates = list(scene_dir.rglob(f"*{band}*10m*.jp2")) or list(scene_dir.rglob(f"*{band}*.jp2"))
        if candidates:
            band_paths[band] = str(candidates[0])

    # Search for SCL (Scene Classification Layer)
    scl_candidates = list(scene_dir.rglob("*SCL*20m*.jp2")) or list(scene_dir.rglob("*SCL*.jp2"))
    scl_path = str(scl_candidates[0]) if scl_candidates else None

    return band_paths, scl_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 Preprocessing Ingestion Pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw scenes directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed arrays")
    parser.add_argument("--crop-size", type=int, default=1024, help="Crop window height and width (e.g. 512, 1024, 2048)")
    parser.add_argument("--training-scene-id", type=str, default=None, help="Specific scene ID to fit standardization on")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Notice: Raw directory {raw_dir} does not exist yet. Create it and place downloaded SAFE granules inside.")
        sys.exit(0)

    # Find potential scene subdirectories
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"No scene directories found in {raw_dir}. See data/README.md for download instructions.")
        sys.exit(0)

    print(f"Scanning {len(subdirs)} scene directories in {raw_dir}...")
    scenes = {}

    for d in subdirs:
        band_paths, scl_path, metadata_path = find_scene_assets(d)
        if len(band_paths) == 4 and scl_path and metadata_path:
            print(f"Found complete Sentinel-2 asset set for {d.name}")
            scenes[d.name] = (band_paths, scl_path, metadata_path)
        else:
            print(f"Skipping {d.name}: missing bands (found {list(band_paths.keys())}), SCL={bool(scl_path)}, XML={bool(metadata_path)}")

    if not scenes:
        print("No complete Sentinel-2 scenes identified. Check file naming or download complete granules.")
        sys.exit(0)

    # Process scenes
    window = (0, 0, args.crop_size, args.crop_size)
    loaded_scenes = {}
    for name, (band_paths, scl_path, metadata_path) in scenes.items():
        print(f"Loading {name} (crop window={window})...")
        scene = load_sentinel2(band_paths, scl_path, metadata_path, window=window)
        loaded_scenes[name] = scene
        print(f"  Valid pixels: {scene.X.shape[0]} / {window[2]*window[3]} ({100*scene.X.shape[0]/(window[2]*window[3]):.1f}%)")

    # Fit standardization statistics on training scene
    train_name = args.training_scene_id if args.training_scene_id in loaded_scenes else list(loaded_scenes.keys())[0]
    print(f"Fitting standardization statistics on training scene: {train_name}")
    train_scene = loaded_scenes[train_name]
    stats = fit_standardization([train_scene.X], training_ids=[train_name], split="train")

    stats_path = out_dir / "standardization_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats.to_json())
    print(f"Saved standardization statistics to {stats_path}")

    # Standardize and save processed arrays
    for name, scene in loaded_scenes.items():
        normalized = scene.standardized(stats)
        scene_out = out_dir / name
        scene_out.mkdir(parents=True, exist_ok=True)

        np.save(scene_out / "X.npy", normalized.X)
        np.save(scene_out / "valid_mask.npy", normalized.valid_mask)
        np.save(scene_out / "pixel_indices.npy", normalized.pixel_indices)

        with open(scene_out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(normalized.metadata, f, indent=2)

        print(f"Exported processed arrays for {name} -> {scene_out}")

    print("Preprocessing ingestion pipeline complete.")


if __name__ == "__main__":
    main()
