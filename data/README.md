# Sentinel-2 Level-2A Dataset Acquisition & Provenance

This directory documents the acquisition, storage policy, and processing pipeline for authentic Copernicus Sentinel-2 Level-2A imagery.

## 1. Storage & Git Policy
- **`data/raw/`**: Stores downloaded raw ESA Sentinel-2 `.SAFE` granules or JP2 assets. **Never committed to Git** (enforced by `.gitignore`).
- **`data/processed/`**: Stores cropped, masked, and standardized `.npy` / `.tif` arrays. Large binary files are **gitignored**.
- **Manifests & Statistics**: Only lightweight JSON metadata manifests and standardization statistics (`stats.json`) are committed.

## 2. Recommended Scene Selection (Copernicus Data Space Ecosystem)
Download cloud-free (< 5% cloud cover) Level-2A products from [Copernicus Browser](https://browser.dataspace.copernicus.eu/):

| Split | Scene Category | Geographic Area | Recommended Tile |
|---|---|---|---|
| **Development (Train)** | Agricultural / Mixed Land-Cover | Central California Valley, USA | `T10SEH` or `T10SFH` |
| **Tuning (Validation)** | Urban / River Delta | Rhine Delta, Germany/Netherlands | `T31UET` or `T32ULB` |
| **Held-out Test** | Coastal / Semi-arid Terrain | Mediterranean Basin, Spain | `T30TXM` or `T30SXJ` |

## 3. Required Spectral Bands & Resolution
The initial baseline extracts **$D=4$ spectral bands** at native 10 m resolution:
- **B02** (Blue, 490 nm, 10 m)
- **B03** (Green, 560 nm, 10 m)
- **B04** (Red, 665 nm, 10 m)
- **B08** (Near Infrared / NIR, 842 nm, 10 m)

Along with the **Scene Classification Layer (SCL)** at 20 m (nearest-neighbor resampled to 10 m common grid):
- Excluded SCL classes: `0` (NoData), `1` (Saturated/Defective), `3` (Cloud Shadow), `8`/`9` (Cloud Medium/High), `10` (Cirrus).
- Retained SCL classes: `2` (Dark), `4` (Vegetation), `5` (Not Vegetated), `6` (Water), `7` (Unclassified), `11` (Snow/Ice).

## 4. Preprocessing Script
Once granules are downloaded into `data/raw/`, run:

```bash
python scripts/prepare_sentinel2_scenes.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --crop-sizes 512 1024 2048
```

This ensures:
1. Standardization statistics ($\mu, \sigma^2$) are fitted **strictly on the development (training) scene** and applied unchanged to validation and test scenes to prevent data leakage.
2. Only valid, cloud-free pixels are included in the 2D feature matrix `X` ($[N, 4]$ float32).
3. Coordinate reference system (CRS), affine transforms, and `pixel_indices` are preserved to reconstruct 2D spatial cluster maps.
