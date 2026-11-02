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
    """Require a single product/granule and unambiguous matching asset names.

    Tile metadata cannot substitute for product calibration XML. Multiple
    granules must be prepared separately rather than mixing first matches.
    """
    import re
    from pcap.preprocessing import read_product_metadata
    scene_dir = scene_dir.resolve()
    xmls = list(scene_dir.rglob("MTD_MSIL2A.xml"))
    if len(xmls) != 1 or xmls[0].parent != scene_dir:
        raise ValueError("Exactly one product-root MTD_MSIL2A.xml is required")
    calibration = read_product_metadata(xmls[0])
    if scene_dir.name.endswith(".SAFE") and scene_dir.name != calibration.product_id:
        raise ValueError("Product directory and calibration product ID disagree")
    paths = {}
    prefixes = set()
    granules = set()
    for band, resolution in [(b, "10m") for b in ("B02", "B03", "B04", "B08")] + [("SCL", "20m")]:
        candidates = [p for p in scene_dir.rglob("*.jp2")
                      if re.fullmatch(r"(.+)_" + band + "_" + resolution, p.stem)]
        if len(candidates) != 1:
            raise ValueError(f"Require exactly one {band} {resolution} asset; found {len(candidates)}")
        asset = candidates[0]
        prefixes.add(asset.stem.rsplit("_", 2)[0])
        ancestors = [p for p in asset.parents if p.parent.name == "GRANULE"]
        if len(ancestors) != 1 or scene_dir not in ancestors[0].parents:
            raise ValueError("Assets must belong to a product GRANULE")
        granules.add(ancestors[0])
        paths[band] = str(asset)
    if len(prefixes) != 1 or len(granules) != 1:
        raise ValueError("Band/SCL assets must share tile, acquisition and granule")
    return {b: paths[b] for b in ("B02", "B03", "B04", "B08")}, paths["SCL"], str(xmls[0])


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 Preprocessing Ingestion Pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw scenes directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed arrays")
    parser.add_argument("--crop-size", type=int, default=1024, help="Crop window height and width (e.g. 512, 1024, 2048)")
    parser.add_argument("--row", type=int, default=0, help="Crop row offset on the 10 m grid")
    parser.add_argument("--column", type=int, default=0, help="Crop column offset on the 10 m grid")
    parser.add_argument("--training-scene-id", type=str, required=True, help="Explicit directory name of the training scene")
    parser.add_argument("--split-manifest", type=Path, required=True, help="JSON mapping scene directory names to train/validation/test")
    args = parser.parse_args()
    if args.row < 0 or args.column < 0:
        parser.error("crop offsets must be nonnegative")
    if args.crop_size < 1:
        parser.error("crop-size must be positive")
    splits = json.loads(args.split_manifest.read_text())
    if not isinstance(splits, dict) or any(v not in ("train", "validation", "test") for v in splits.values()):
        parser.error("Invalid split manifest")
    if splits.get(args.training_scene_id) != "train":
        parser.error("training-scene-id must be explicitly assigned to train")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"Notice: Raw directory {raw_dir} does not exist yet. Create it and place downloaded SAFE granules inside.")
        raise ValueError("No usable scene inputs")

    # Find potential scene subdirectories
    subdirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if set(splits) != {p.name for p in subdirs}:
        raise ValueError("Split manifest must name exactly the input scene directories")
    if not subdirs:
        print(f"No scene directories found in {raw_dir}. See data/README.md for download instructions.")
        raise ValueError("No usable scene inputs")

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
        raise ValueError("No usable scene inputs")

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Choose a new empty output directory; prepared scenes are never overwritten")
    # Process scenes
    window = (args.row, args.column, args.crop_size, args.crop_size)
    loaded_scenes = {}
    for name, (band_paths, scl_path, metadata_path) in scenes.items():
        print(f"Loading {name} (crop window={window})...")
        scene = load_sentinel2(band_paths, scl_path, metadata_path, window=window, max_crop_pixels=args.crop_size ** 2)
        loaded_scenes[name] = scene
        print(f"  Valid pixels: {scene.X.shape[0]} / {window[2]*window[3]} ({100*scene.X.shape[0]/(window[2]*window[3]):.1f}%)")

    # Fit standardization statistics on training scene
    train_name = args.training_scene_id
    if train_name not in loaded_scenes:
        raise ValueError("Requested training scene was not loaded")
    if [name for name, split in splits.items() if split == "train"] != [train_name]:
        raise ValueError("This pipeline requires exactly one explicit training scene")
    out_dir.mkdir(parents=True, exist_ok=True)
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
        normalized.metadata.update(scene_id=name, split=splits[name],
                                   source_kind="caller_supplied_product_files")
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
