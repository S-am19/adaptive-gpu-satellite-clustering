# Genuine Sentinel-2 workflow — 24 October 2026

This report supersedes the pre-acquisition limitations in `RESULTS_STATUS.md`
**for one genuine SAFE product**. It does not establish cross-scene generalization
or complete the GPU ANN/combined objective.

## Product and integrity

- Product: `S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE`.
- Sentinel-2A Level-2A, tile 31UFS, acquired 25 June 2024; metadata datatake start
  `2024-06-25T10:36:31.024Z`; baseline 05.10.
- User downloaded the original ZIP to `C:\Projects\PCAP\`; it was copied to
  WSL `data/raw/20240625-31UFS/`, without changing the Windows original.
- Archive: 1,238,257,912 bytes; 95 entries; all ZIP CRC checks passed.
- SHA256: `0a7e17080bbdb9cb04f3fa0b3598b79aeacaa6c18e282d17c3eea9abfff7b924`.
- All required image assets are genuine `JP2OpenJPEG` rasters. B02/B03/B04/B08
  are uint16 10980×10980, common EPSG:32631 grid at 10 m. SCL is uint8
  5490×5490 at 20 m in the same CRS.
- Actual product XML specifies quantification 10000 and offsets -1000 for all
  selected bands. DN=0 and saturated DN=65535 are excluded before calibration.
  XML SHA256: `c456a8ece6aad8fa42e95f8ac1755183c497f8d2d730c9a03e1100a8c23cf8ec`.

These are original raw product DN, not synthetic payloads or already offset-
corrected COGs. Catalogue cloud cover is 0.81105%, but crop-level masks are used.

## Prepared development crop

Maastricht-area 512×512 window: `(row=6084, column=8679, height=512, width=512)`.
The odd column offset exercises nonaligned 20 m SCL resampling. Output origin is
E=686790, N=5639160, EPSG:32631, 10 m pixels. All 262144 crop pixels are eligible;
SCL counts: class 2=1088, 4=154544, 5=96984, 6=9292, 7=232, 11=4. Snow class 11
is retained under the documented default; these four SCL labels are not evidence
of actual snow. Cloud/shadow/no-data categories do not occur in this crop.

Sampled reflectance values were independently checked against `(DN-1000)/10000`.
An independent GDAL nearest-neighbor reprojection verified SCL alignment. Finite,
contiguous float32 `[262144,4]` features use B02/B03/B04/B08 order. Standardization
was fitted once on this development crop; it is not fitted on later crops.
Feature SHA256: `1e1c47e713ec4a5cf8f9e47303efe59ae3947cec22dc774024e3e198a5d3d56a`.

A 256×256 Maastricht subcrop uses window `(6212,8807,256,256)` and reuses these
statistics unchanged. A separate Hasselt-area 512×512 window `(5281,6117,512,512)`
also reuses the statistics. It is a **spatial test within the same product**, not
an independent held-out acquisition/scene. Both yield all eligible pixels.

## First real all-method smoke evaluation

`outputs/20261024-real-smoke-512/`: K=4, seed=0, batch=512, max_steps=30,
tolerance=1e-5, one warmup and one measured fit per method. All nine methods
passed exact final-label/SSE gates; all reached the update cap. Single-fit times
are smoke observations, not reliable speedup estimates.

Exact CPU/GPU final SSE: 328263.774522 (difference about 4.2e-9). Fixed and
adaptive Mini-Batch variants in this smoke run had final SSE 407803.822843;
therefore their lower work/time is not a matched-quality speedup.

The G0 map was reconstructed and independently checked for EPSG, affine transform,
512×512 dimensions and 262144 assigned pixels. True-color B04/B03/B02 displays
show city, river, vegetation and fields; colors use display-only percentile
stretches. Cluster labels remain unsupervised IDs, not verified land-cover classes.

## Output locations and reproduction

- Integrity/calibration checks: `outputs/20261024-real-scene-validation/`.
- Original archive/product: `data/raw/20240625-31UFS/` (ignored by Git).
- Prepared crops: `outputs/20261024-real-maastricht-512/`,
  `outputs/20261024-real-maastricht-256/`, `outputs/20261024-real-hasselt-512/`.
- Initial real models/maps: `outputs/20261024-real-smoke-512/`; view
  `maps-G0/true_color.png`, `false_color_nir.png`, `clusters.png`, `clusters.tif`.

```bash
.venv/bin/python -B scripts/prepare_sentinel2_scenes.py \
  --raw-dir data/raw/20240625-31UFS --output-dir outputs/real-prepared-NEW \
  --training-scene-id S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE \
  --split-manifest outputs/20261024-real-scene-validation/splits.json \
  --crop-size 512 --row 6084 --column 8679
MPLCONFIGDIR=/tmp/pcap-mplconfig timeout 150s bash scripts/run_suite_cuda.sh \
  --scene-dir outputs/real-prepared-NEW/S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE \
  --methods C0 C1 G0 G1 G2 G3 G4 G5 G3CUDA --k 4 --seeds 0 \
  --batch-size 512 --max-steps 30 --warmups 1 --repeats 1 \
  --budget-seconds 120 --save-models --output-dir outputs/real-smoke-NEW
```

Rendering and plotting commands are in the README. Use newly named output
folders; previous archives/models/maps must never be overwritten.

## Repeated real-data evaluation

The 30-update matrix used two nested crop sizes (65536 and 262144 valid points),
D=4, K=4/16, seeds 0/1, initial batch 512, two warmups and five measured repeats
per method/case. C0/C1/G0/G1/G2/G3/G4/G5/G3CUDA shared identical X and C0,
K, seed, step cap and tolerance within each case. Batch-size and ANN-effort
configuration fields were method-specific; equal steps do not imply equal work. This produced **360 measured fits**, all with exact final
label/SSE checks. All terminated at max_steps=30; these are capped-fit results,
not evidence of convergence or matched-quality Mini-Batch speedups.

One original 512×512 run hit its 120-second budget between fits after three
fully completed cases (135 measurements). Its failure/partial warmups were
preserved. Only the remaining K=16, seed=1 case was completed in a new process.
Two smaller-crop batches were also separate processes. The combined CSV contains
disjoint cases with source-archive columns; each case has five repeats, with no
cross-session pooling of duplicate repetitions. Crops have different pixel
mixtures, so size trends also reflect workload composition and measurement time.

Exact full-fit speedups are CPU median / GPU median:

| N | K | Seed | CPU median ms | GPU median ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 65536 | 4 | 0 | 47.59 | 242.05 | 0.197x |
| 65536 | 4 | 1 | 47.56 | 260.09 | 0.183x |
| 65536 | 16 | 0 | 162.93 | 292.57 | 0.557x |
| 65536 | 16 | 1 | 163.64 | 299.75 | 0.546x |
| 262144 | 4 | 0 | 184.15 | 358.54 | 0.514x |
| 262144 | 4 | 1 | 183.28 | 420.02 | 0.436x |
| 262144 | 16 | 0 | 648.49 | 527.60 | 1.229x |
| 262144 | 16 | 1 | 650.99 | 560.73 | 1.161x |

CPU/GPU exact steps, termination and SSE matched. GPU was faster only at the
larger N/K=16 here. Individual timings, IQR and outliers are retained; these
measurements do not establish universal acceleration or explain clock/thermal/
scheduling variation. Mini-Batch processed 15360 observations versus up to
7864320 for full Lloyd at the larger size, and generally had higher SSE.

Derived comparisons are in `outputs/20261024-real-evaluation/`:
`measurements.csv`, `sources.json`, `scaling/median_summary.csv`,
`scaling/exact_speedups.csv`, scaling plots and quality/throughput/ablation figures.
The incomplete archive is explicitly marked in `sources.json`.

## Actual real-data adaptation diagnostic

Separate protocol: N=65536, D=4, K=4, seed=0, initial batch=512, max_steps=300,
two warmups/five measured repetitions per method. Do not combine it with the
30-update study. Exact CPU/CUDA both satisfied tolerance after 55 Lloyd updates
with SSE 75316.992940; Mini-Batch methods all reached 300 updates.

| Method | Median full fit ms | Final exact SSE | Observations |
|---|---:|---:|---:|
| C0 exact CPU | 83.542 | 75316.992940 | full Lloyd, 55 updates |
| G0 exact CUDA | 463.926 | 75316.992940 | full Lloyd, 55 updates |
| C1 fixed CPU Mini-Batch | 40.158 | 78255.726029 | 153600 |
| G1 fixed CUDA Mini-Batch | 2156.724 | 78255.726029 | 153600 |
| G2 fixed CPU HNSW | 260.015 | 78255.726029 | 153600 |
| G3 adaptive CPU batch | 43.768 | 78279.747366 | 147200 |
| G3CUDA adaptive CUDA batch | 1997.757 | 78279.747366 | 147200 |
| G4 adaptive CPU ANN effort | 266.202 | 78255.726029 | 153600 |
| G5 combined CPU | 264.305 | 78279.747366 | 147200 |

The batch-size decision at checkpoint 275 applies from update 276 (512→256).
G4 reduces effort 64→32 from update 276. G5 changes both at update 276 and reduces
ANN effort again to 16 at update 291. GPU/CPU batch schedules agree. Batch
adaptation saved 6400 observations but had slightly higher SSE and was slower
on CPU. ANN/combined CPU adaptation did not beat fixed CPU ANN. High-effort
HNSW at K=4 often returns exact neighbors; reduced effort is not evidence of
fewer distance evaluations or meaningful approximation errors. The code remains
a genuine HNSW backend, with maintenance included, not a disguised exact fallback.
No adaptive speedup at a predeclared target quality is established.

The spatial Hasselt test completed all nine methods with unchanged development
normalization. Its single measured fits validate execution/maps only. The separate
cloud-boundary crop `(0,4069,256,256)` excludes 17128 pixels: SCL shadow=7356,
medium cloud=8244, high cloud=1528. The remaining 48408 pixels yield finite features.
This confirms actual cloud/shadow masking on genuine imagery, not just fixtures.

Across the real archives: **423 measured fits and 192 warmups** completed, plus
the one explicitly recorded budget stop. No clustering correctness failure was
hidden. Raw CSV/JSONL, source snapshots, saved models and checksums remain local
outside Git. Maps include exact CUDA K=4/K=16, combined CPU and adaptive CUDA,
plus the separate spatial-test map; all preserve the selected crop geometry.

## Limits still outstanding

- This is one genuine acquisition; the spatial test is not an independent scene.
- G2/G4/G5 remain CPU prototypes. GPU ANN and joint GPU adaptation are not done.
- Longer/larger multi-session held-out experiments and time-to-target-quality
  curves are still needed. Current quality plots use final full-data SSE.
- No supervised land-cover ground truth is available; cluster IDs are not classes.
- OpenMP and the original CUDA C++/Python scope require instructor resolution.
- No global environment/driver changes, pushes or merges were performed.

## Final verification

After the real workflows, the CPU suite again passed **343 tests** (82 warnings)
and the full real-GPU suite passed **393 tests** (101 warnings), with RTX 4050 and
CUDA simulator disabled. `pip check` and whitespace checks passed. No algorithm
or dependency changes were needed to process this product.

The original inventories of all seven experiment archives and their source
hashes were verified. A master SHA256 inventory verifies **1587 real-output
files**, including prepared arrays, source snapshots, models, measurements,
plots and provenance. From the repository root:

```bash
sha256sum -c outputs/20261024-real-evaluation/checksums-all-real.sha256
```

View `outputs/20261024-real-evaluation/real-scene-comparison.png` for the real
true-color/NIR/cluster comparison. The working branch remains integration-daksh;
the validated implementation is local commit a48eab8; this report records the
subsequent real-data evidence. No push or merge was performed.

## Acceptance against the unchanged original plan

The current deliverable is a validated real-data research prototype, not full
completion of the original matrix. The plan specifies GPU G2/G4/G5 (fixed ANN,
adaptive ANN and joint adaptation); CPU implementations with these IDs are
explicitly labeled prototypes, and G3CUDA supplies the GPU batch-only variant.
M5 allows an honest negative ANN feasibility finding and discussion of scope
with the instructor. That is a decision gate, not automatic permission to waive
the GPU variants. Implement and validate the remaining variants, or obtain and
record an approved scope adjustment; no such approval is claimed here.

Remaining evaluation coverage includes separate development/tuning/untouched
scene groups, frozen settings and time-to-target-quality evaluation, additional
real sizes (1024/2048-square where feasible), K=8/32 in the required real matrix,
and the planned seed/session coverage. Five repetitions here are not five
independent random seeds; the real comparison uses two seeds. OpenMP deferral
and Python-defined CUDA acceptance also require the instructor decision stated
in M8. A successful real map and passing tests do not waive these requirements.
