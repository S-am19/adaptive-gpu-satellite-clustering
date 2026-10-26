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

## Limits in the original 24 October report

- This is one genuine acquisition; the spatial test is not an independent scene.
- At that report date, G2/G4/G5 were CPU prototypes and GPU ANN/joint adaptation
  had not yet been implemented. See the continuation below for later work.
- Longer/larger multi-session held-out experiments and time-to-target-quality
  curves are still needed. Current quality plots use final full-data SSE.
- No supervised land-cover ground truth is available; cluster IDs are not classes.
- OpenMP had not yet been implemented; the original CUDA C++/Python scope
  required instructor resolution.
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

The original report's coverage statements above describe work through 24
October. The continuation below supersedes the GPU ANN/OpenMP implementation
status but does not complete the full plan matrix or instructor decision.

## Continuation: GPU ANN, quality target and OpenMP — 25 October 2026

This work is kept separate from the earlier real-scene measurements.

### GPU projection-ANN feasibility

`outputs/20261025-projection-feasibility-final/` contains the final-source
synthetic RTX 4050 study: N=8192, D=4/14, K=4/8/16/32/64/128/256, seed 0,
effort 1/4/16, two warmups and five measured queries per method/case. The 392
measured rows include exact GPU search and custom sorted random-projection
candidate search. The archive has 392 raw queries (112 warmups and 280
measured). ANN results are re-scored in float64 and compared against exact CPU
assignments; complete timing includes index/centroid transfer, query and result
retrieval. Source snapshots and recursive checksums are preserved.

In every genuine approximate mode, median complete cost was higher than exact
GPU search. Effort 1 recall ranged from 0.8% to 16.6% across tested D/K cases;
low-effort objective gaps could be substantial. K<=8 uses explicitly exact
search, and effort>=K is explicitly exhaustive. The pilot supports a negative
feasibility finding for this projection design and tested range, not a GPU ANN
speedup claim.

### Frozen-target real-image quality pilot

`outputs/20261025-real-quality-256-k16/` uses 65,536 standardized pixels from
the genuine 31UFS product, D=4, K=16, seed=0, max_steps=30, tolerance=1e-6,
two warmups, five measured fits and full-data quality checks every five updates.
The converged C0 reference SSE was 18,650.758683; the frozen target was
19,583.296618 (5% above reference). C0 and G0 each reached the target in 5/5
fits, with median instrumented complete times of 152.45 ms and 42.79 ms.
G1/G2CUDA/G3CUDA/G4CUDA/G5CUDA reached it in 0/5 and stopped at the 30-step
cap. Those methods are target misses, not successful speedups. The trajectory,
target summary, raw JSONL, source snapshot and recursive checksums are preserved.

This is one crop, K and seed. It does not establish adaptive benefit, validation
quality or held-out performance. These instrumented times cannot be pooled with
the legacy real-study medians above.

### C++ OpenMP real-image thread pilot

`outputs/20261025-openmp-real-maastricht-256/` compares C0 with native C++ O0
on identical N=65,536, D=4, K=4, seed=0 pixels and initial centroids. Every run
used 55 updates, two warmups and five measured fits. Median external O0 times at
1/2/4/8 threads were 83.57/54.74/44.35/72.58 ms. The corresponding per-session
C0/O0 ratios were 1.01x/1.54x/1.88x/1.20x. `native_fit_s` separates C++ core
time from process startup and serialization. The result is one small-crop
session, not a general thread-scaling claim.

### Current acceptance gates

- Three genuine acquisitions are now available: development 31UFS, validation
  31UFU and test 30STG. Their 512×512 validation/test crops use the same frozen
  development statistics. Archive/XML checksums and feature hashes are in
  `data/independent_scene_candidates.json` and
  `outputs/20261025-independent-holdouts/`. Exact K=16 CPU/GPU baseline maps
  exist for both independent crops. The test crop has not been used to select
  adaptive settings.
- The real validation policy is frozen, and all twelve methods have five-seed
  held-out results at K=16. The full real N/K/D matrix, held-out test results
  for other K values, and independent repeat sessions for every variant remain
  incomplete. Preserve capped runs, quality misses and failures; do not tune on
  the test crop.
- The installed WSL Nsight Systems did not produce a usable CUDA timeline. Per-
  kernel bottlenecks are therefore not established; profiler work was not resumed.
- Instructor confirmation that Python-authored CUDA meets the proposal's C/C++
  language requirement is still open. The C++ OpenMP baseline does not answer
  that question.

After the 25 October source/test additions, **368 CPU-side tests** and **438
real-GPU tests** passed. The complete GPU command was
`bash scripts/run_cuda_tests.sh tests -x`; the CPU-only command excludes
`tests/test_cuda_ann.py` as well as the other hardware test modules. See
[`IMPLEMENTATION_PLAN_STATUS.md`](IMPLEMENTATION_PLAN_STATUS.md) for the stage
checklist and remaining approval gates.

### Independent validation and held-out test continuation — 25 October 2026

The validation crop is genuine 31UFU, N=262,144, D=4, using the frozen
Maastricht development standardization. All results here are K=16, seed=0 and
separate from the earlier 31UFS study. Raw measurements, source snapshots and
plots are under the respective `outputs/20261025-validation-*` directories;
recursive inventories named `checksums_all.sha256` include the files below
each study directory.

An exact reference pair on 31UFU converged at update 138, with identical CPU/GPU
labels and centroids and SSE 24,395.035. Its one measured run per method is for
map production and correctness; do not use its time for a runtime claim. The
aligned outputs are in `outputs/20261025-validation-scene-k16-exact/render_C0/`
and `render_G0/`.

The first validation candidate used 8,192-point batches, max_steps=100,
tolerance=1e-6, two warm-ups/five measured fits, full-data checks every 10
updates, and a frozen target 5% above the converged exact reference (SSE
24,395.035; target 25,614.786). C0 and G0 reached the target in all five
measured fits. Every Mini-Batch method reached it in zero of five and terminated
at max_steps; adaptive decisions remained `active`.

Two fixed Mini-Batch validation continuations used max_steps=1,000, two
warm-ups/five measurements and full-data checks every 50 updates. At batch
8,192, CPU and GPU produced identical final SSE 30,249.802, each missed the
target 0/5 and reached the step cap. Median instrumented fit times were 1.851 s
(CPU, IQR 0.034 s) and 5.986 s (GPU, IQR 4.392 s). At batch 65,536, both again
reached the cap with identical SSE 30,380.620 and zero target hits; medians were
7.943 s (CPU, IQR 0.161 s) and 10.631 s (GPU, IQR 0.439 s). These are
single-seed tuning pilots. CUDA timing variability was large in the 8,192 run;
times from separate sessions do not establish a speedup. The larger batch did
not improve quality or runtime in this protocol.

A separate G3CUDA validation check used batch 8,192, max_steps=1,000, two
warm-ups/five measurements and checks every 50 steps. All five runs stopped at
max_steps with SSE 30,249.802 and missed the target. Each had 100 `active`
controller checkpoints and no batch-size transitions. This does not
demonstrate an adaptive batch benefit or successful convergence.

The 30STG test crop received an exact CPU/GPU comparison at N=262,144, D=4,
K=16, seed=0 and 300 updates, with two warm-ups/five measured fits. Both methods
had SSE 32,489.433 and identical labels/centroids, but both stopped at
max_steps without satisfying tolerance. Median external times were 5.831 s
CPU (IQR 0.074 s) and 1.500 s CUDA (IQR 0.904 s). This is a single fixed-work
session; its observed 3.89x median ratio is not generalized. A separate
max_steps=1,000 run reached tolerance on both methods at update 406; both
returned SSE 32,485.553 and identical labels/centroids. It had one measured fit
per method and is not a stable timing comparison.

Test-scene map outputs are in
`outputs/20261025-test-scene-k16-exact-cap1000/`: `render_C0/` and `render_G0/`
contain label GeoTIFFs, cluster PNGs, true-color B04/B03/B02 and false-color
B08/B04/B03 views, provenance and checksums. The GeoTIFFs use the exact crop
transform/CRS and nodata=-1. The held-out scene was used for this exact baseline
only; Mini-Batch, ANN and adaptive models have not been evaluated on it and it
must not be used for policy selection.

A frozen K=16 test-quality candidate was then attempted with max_steps=1,000,
batch size 8,192, the validation policy file, a 5% exact-reference target,
checks every 50 updates and two warm-ups/five measured fits. The combined
12-method run exhausted its 120 s budget during warm-ups and recorded zero
measured rows. A separate CPU-only scheduling attempt preserved eight measured
rows (C0: one; C1/G2: two each; G3/G4/G5: one each) before its budget stop. Both
manifests are incomplete and retain the timeout; the partial rows are not a
complete comparison. At that stage no CUDA variant had a measured row. These
initial incomplete artifacts and their frozen protocol remain preserved under
`outputs/20261025-test-quality-k16-frozen-candidate*/`; the completed CPU and
CUDA groups below supersede them for this one frozen K=16 test protocol.

The validation and test reference products are recorded in
`data/independent_scene_candidates.json`, including crop, feature and result
metadata. Each output directory has an independently verified recursive
`checksums_all.sha256` inventory.

The first frozen test-quality seed was completed in two separate execution-
device groups using the same test feature matrix and initial centroids
(matching SHA256 hashes), policy, and 5% target. Each group used two warm-ups
and five measured fits per method, max_steps=1,000, and exact full-data checks
every 50 updates. The converged exact reference SSE was
32,485.552512; the predeclared target was 34,109.830137. The test scene was not
used to alter settings.

| Execution group / method | Target hits | Final exact SSE | Updates / termination | Median fit s (IQR s) |
|---|---:|---:|---|---:|
| CPU C0 exact | 5/5 | 32,551.862 | 250 / quality_target | 5.681 (0.438) |
| CPU C1 fixed mini-batch | 0/5 | 40,449.443 | 1,000 / max_steps | 1.568 (0.163) |
| CPU G2 HNSW | 0/5 | 40,449.443 | 1,000 / max_steps | 11.046 (1.677) |
| CPU G3 adaptive batch | 0/5 | 40,526.302 | 1,000 / max_steps | 1.429 (0.102) |
| CPU G4 adaptive HNSW effort | 0/5 | 40,449.443 | 1,000 / max_steps | 8.910 (1.118) |
| CPU G5 combined | 0/5 | 40,526.302 | 1,000 / max_steps | 8.287 (1.091) |
| CUDA G0 exact | 5/5 | 32,551.862 | 250 / quality_target | 1.457 (0.107) |
| CUDA G1 fixed mini-batch | 0/5 | 40,449.443 | 1,000 / max_steps | 1.221 (0.141) |
| CUDA G2CUDA projection ANN | 0/5 | 41,065.815 | 1,000 / max_steps | 7.712 (0.439) |
| CUDA G3CUDA adaptive batch | 0/5 | 40,526.302 | 1,000 / max_steps | 6.042 (0.069) |
| CUDA G4CUDA adaptive projection effort | 0/5 | 41,187.173 | 1,000 / max_steps | 10.136 (0.978) |
| CUDA G5CUDA combined | 0/5 | 40,693.737 | 1,000 / max_steps | 11.509 (0.508) |

`quality_target` means the 5% objective threshold was reached; it does not mean
the K-Means tolerance criterion was satisfied. All other methods were capped,
not converged. Only exact C0/G0 reached the target. The adaptive and ANN variants
therefore do not establish a matched-quality improvement on this crop. Each
method's five individual timings and all quality checkpoints are retained in
the group's `measurements.csv` and `raw.jsonl`. Timings are instrumented
complete-fit measurements that include full-data quality monitoring. CPU and
CUDA methods were run in separate groups, so their medians must not be divided
to claim a speedup; ANN variants also use different CPU-HNSW and GPU-projection
backends. The full protocol, source snapshots, figures, post-run GPU snapshot
and recursive checksums are in:

- `outputs/20261025-test-quality-k16-frozen-candidate-cpu-complete/`
- `outputs/20261025-test-quality-k16-frozen-candidate-cuda-complete/`

The CUDA run completed 30 measured fits with no recorded failures on the RTX
4050. It emitted Numba low-occupancy warnings for some small launch grids. The
runner's pre-run snapshot recorded P0, 2055 MHz and 46 C; an immediate manual
pre-run query read 45 C, and a separate post-run snapshot read P0, 2055 MHz and
45 C. These snapshots are not per-fit telemetry and do not explain run-to-run
outliers. The frozen policy missed the validation target before this held-out
evaluation, and this test result does not justify retuning it.

The post-campaign hardware snapshot is in
`outputs/20261025-validation-and-test-followup/post_campaign_hardware.json`.
It is labeled as a later snapshot, not per-fit telemetry.

### Five-seed held-out replication

The same frozen protocol was then run for seeds 0–4 across all twelve methods,
with five measured repetitions per method and two warm-ups. For each seed, CPU
and CUDA were run in separate groups. X hashes match across seeds; within each
seed, CPU and CUDA used the same C0 hash, exact reference, and target. No
settings were changed after inspecting the test scene. All ten run archives
contain raw per-fit records, manifests, source snapshots, and verified
recursive `checksums_all.sha256` inventories. The combined artifact is
`outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/`; it includes per-
seed and aggregate CSVs plus ten seed-specific objective trajectory plots.
Rebuild it from the ten run archives with:

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B experiments/summarize_holdout_quality.py \
  --seed-pair outputs/20261025-test-quality-k16-frozen-candidate-cpu-complete outputs/20261025-test-quality-k16-frozen-candidate-cuda-complete \
  --seed-pair outputs/20261025-test-quality-k16-seed1-cpu outputs/20261025-test-quality-k16-seed1-cuda \
  --seed-pair outputs/20261025-test-quality-k16-seed2-cpu outputs/20261025-test-quality-k16-seed2-cuda \
  --seed-pair outputs/20261025-test-quality-k16-seed3-cpu outputs/20261025-test-quality-k16-seed3-cuda \
  --seed-pair outputs/20261025-test-quality-k16-seed4-cpu outputs/20261025-test-quality-k16-seed4-cuda \
  --output-dir outputs/heldout-quality-summary-NEW
```

Across 25 measured fits per method and execution group, exact C0/G0 reached the
seed-specific 5% target in 25/25 fits. Every fixed Mini-Batch, ANN, batch-
adaptive, ANN-adaptive, and combined method reached `max_steps=1,000` in 0/25
fits. The median final exact SSE was 32,615.653 for C0/G0, 39,672.467 for
C1/G1 and CPU HNSW G2/G4, 39,710.766 for batch-adaptive G3/G5/G3CUDA,
39,646.540 for CUDA projection G2CUDA, 39,961.256 for G4CUDA, and 39,655.691
for G5CUDA. Target status is distinct from Lloyd tolerance convergence; the
non-exact methods were capped, not converged.

The aggregated medians of complete instrumented fit time were 5.510 s (IQR
1.143 s) for CPU C0 and 1.442 s (IQR 0.225 s) for CUDA G0. The CPU and CUDA
groups were separate sessions, so these are descriptive timings and must not
be divided into a speedup. CUDA G1 had substantial timing variability (IQR
1.950 s versus a 1.300 s median); all repetitions remain in the raw archive.
The CUDA runs emitted Numba low-occupancy warnings for small grids. Hardware
snapshots are session-level, not per-fit telemetry.

### Four-K validation quality sweep and real-image N scaling — 25 October 2026

The validation K sweep uses the genuine 31UFU crop (N=262,144 valid pixels,
D=4) with the same feature hash, five seeds, five measured repeats per method,
two warm-ups, batch size 8,192, max_steps=100, evaluation/probe intervals 10,
and the policy frozen before the held-out evaluation. Each K has a distinct
archive and per-seed target plot; the combined summary is
`outputs/20261025-validation-quality-k-sweep-summary-v2/`.

| K | Exact C0/G0 target hits | Non-exact methods' target hits | Median paired exact C0/G0 CPU/CUDA ratio (IQR) |
|---:|---:|---:|---:|
| 4 | 25/25 | 15/25 each | 0.744× (0.341×) |
| 8 | 25/25 | 0/25 each | 0.930× (0.204×) |
| 16 | 25/25 | 0/25 each | 1.363× (0.211×) |
| 32 | 25/25 | 0/25 each | 3.283× (1.536×) |

The target is 5% above a separately converged exact reference for each seed.
`quality_target` is not the Lloyd tolerance criterion. Mini-Batch/ANN/adaptive
variants generally reached the 100-update cap. C0/G0 ratios are reported only
after the summary script checks matching X/C0 hashes, updates, termination,
target status, and final SSE. Ratios are per-K, one-session time-to-target
observations, not a universal speedup or evidence that adaptation helped.

The real fixed-work scaling extension used nested 31UFS Maastricht crops,
B02/B03/B04/B08 and unchanged training-only standardization, K=16, five seeds,
two warm-ups, and five measured fits per method. It included O0, C0/G0, fixed
Mini-Batch, CPU HNSW, and CPU/CUDA adaptive/combined variants. Every fit made
exactly one update and terminated at `max_steps`; this does not measure
convergence or permit an adaptive transition.

| Valid N | Exact CPU median (IQR) | Exact CUDA median (IQR) | Median paired C0/G0 ratio (IQR) |
|---:|---:|---:|---:|
| 1,048,548 | 0.1487 s (0.0089) | 0.1369 s (0.0076) | 1.087× (0.090×) |
| 4,194,265 | 0.6302 s (0.0297) | 0.2884 s (0.0772) | 2.052× (0.696×) |

The paired exact ratios include the complete synchronized fit, final exact
assignment and CUDA transfers. They are two fixed-work sessions on nested
windows from one product; crop composition and session state limit
generalization. Mini-Batch/ANN times are not divided into CPU/GPU speedups
because those methods do different assignment/update work. Full individual
measurements, hardware/dependency manifests, quality metrics and plots are in
`outputs/20261025-real-fixed-work-N1048548-K16/`,
`outputs/20261025-real-fixed-work-N4194265-K16/`, and
`outputs/20261025-real-fixed-work-scaling-summary-v2/`.

The completed held-out test output maps for C0/C1/G0/G1/G2/G3/G4/G5 and the
CUDA ANN/adaptive variants are under
`outputs/20261025-test-scene-all-method-maps/render_*/`. Each GeoTIFF was
checked against the test crop's CRS, affine transform, nodata mask and saved
labels. `quality_metrics_sampled.csv` records exact final SSE plus silhouette
and Davies–Bouldin metrics on the same seeded 3,000-pixel sample for all twelve
maps. These unlabeled-cluster metrics are not land-cover accuracy. The map
fits use the frozen policy but run to tolerance/max_steps; they are separate
from the target-terminated timing study.

Rebuild the cross-K quality summary from the four complete study folders:

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B experiments/summarize_quality_matrix.py \
  --studies outputs/20261025-validation-quality-k4-frozen-policy \
    outputs/20261025-validation-quality-k8-frozen-policy \
    outputs/20261025-validation-quality-k16-frozen-policy-100 \
    outputs/20261025-validation-quality-k32-frozen-policy \
  --output-dir outputs/quality-k-sweep-NEW
```

Rebuild real fixed-work scaling plots from two complete `run_suite.py` output
directories with:

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B experiments/plot_real_scaling.py \
  --runs outputs/20261025-real-fixed-work-N1048548-K16 \
    outputs/20261025-real-fixed-work-N4194265-K16 \
  --output-dir outputs/real-scaling-NEW
```

Both summaries validate complete rows and exact-work pairs before plotting, and
write checksums for their generated CSVs, manifests and figures.

Re-run the validation K=4/8/16/32 time-to-quality matrix from the already
prepared 31UFU crop (this executes four separate studies and writes new
archives; it does not tune from the test scene):

```bash
for K in 4 8 16 32; do
  timeout 930s bash scripts/run_cuda_python.sh experiments/benchmark_quality.py \
    --scene-dir outputs/20261025-independent-holdouts/validation512 \
    --output-dir "outputs/validation-k${K}-NEW" \
    --methods C0 C1 G0 G1 G2 G3 G4 G5 G2CUDA G3CUDA G4CUDA G5CUDA \
    --k "$K" --seeds 0 1 2 3 4 --max-steps 100 --reference-steps 1000 \
    --tolerance 1e-6 --batch-size 8192 \
    --policy-json configs/holdout_k16_conservative_policies.json \
    --quality-gap 0.05 --evaluation-interval 10 --probe-interval 10 \
    --warmups 2 --repeats 5 --budget-seconds 900
  .venv/bin/python -B experiments/plot_quality.py \
    --study "outputs/validation-k${K}-NEW" \
    --output-dir "outputs/validation-k${K}-NEW/figures"
done
```

Re-run the two larger one-update real-image cases with the same five seeds and
repeats. The prepared arrays are training-scene windows using frozen
Maastricht512 statistics; these are **fixed-work**, not convergence tests:

```bash
for scene in maastricht1024 maastricht2048; do
  timeout 930s bash scripts/run_cuda_python.sh experiments/run_suite.py \
    --scene-dir "outputs/20261024-plan-real-prepared/${scene}" \
    --methods O0 C0 C1 G0 G1 G2 G3 G4 G5 G2CUDA G3CUDA G4CUDA G5CUDA \
    --k 16 --seeds 0 1 2 3 4 --batch-size 8192 --max-steps 1 \
    --tolerance 1e-6 --probe-interval 5 \
    --policy-json configs/default_policies.json --omp-threads 4 \
    --warmups 2 --repeats 5 --budget-seconds 900 \
    --output-dir "outputs/fixed-work-${scene}-NEW"
done
.venv/bin/python -B experiments/plot_real_scaling.py \
  --runs outputs/fixed-work-maastricht1024-NEW \
    outputs/fixed-work-maastricht2048-NEW \
  --output-dir outputs/real-scaling-NEW
```

### Development N/K quality matrix completion — 26 October 2026

This extension fills the 31UFS development-scene N/K cells at the prepared
1024×1024 and 2048×2048 Maastricht windows. It uses B02/B03/B04/B08 (D=4),
the existing Maastricht512 training normalization, seeds 0–4, initial batch
8192, frozen conservative policies, max_steps=100, tolerance 1e-6, and a
per-seed converged exact CPU reference with a frozen 5% allowance. Full-data
quality is checked every 10 updates. No parameters or thresholds were changed
from test-scene observations.

| Valid N | K | Exact C0/G0 target hits | G1 fixed batch | G2CUDA fixed ANN | G3CUDA adaptive batch | G4CUDA adaptive ANN | G5CUDA joint |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,548 | 4 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1,048,548 | 8 | 10/10 | 2/10 | 2/10 | 2/10 | 2/10 | 2/10 |
| 1,048,548 | 16 | 6/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 1,048,548 | 32 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 4,194,265 | 4 | 5/5 | 3/5 | 3/5 | 3/5 | 3/5 | 3/5 |
| 4,194,265 | 8 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 4,194,265 | 16 | 3/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 4,194,265 | 32 | 3/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

N=1,048,548 K=4/8/16 has two measured repeats per seed; other listed cells
have one measured fit per seed. Warm-ups are excluded. The two 4-million-point
K=16/32 runs were split after timeout-limited processes: all five seed records
are present exactly once across their partial raw archives and seed continuation
archives. Their source snapshots and raw files are listed in the combined
manifest. Paired C0/G0 X/C0 hashes, target SSE, stopping fields and final SSE
were checked before the combined CSV was generated.

The max_steps=100 cap is a material limit. Fixed batch 8192 processes at most
819,200 observations in 100 updates: about 0.78 passes over the 1.05M crop and
0.20 passes over the 4.19M crop. Most Mini-Batch/ANN/adaptive rows ended at
`max_steps`; target misses are not reported as convergence. C0/G0 also missed
the target for some K=16/32 seeds at N≥1.05M, so those cells do not support a
complete time-to-common-quality comparison. At N=1.05M K=4/8 and N=4.19M
K=4/8, all paired exact runs reached the target; their paired median C0/G0
ratios were respectively 1.387× (IQR 0.392×), 2.484× (0.997×), 3.968×
(0.699×), and 5.624× (0.859×). These are one-session end-to-end observations,
not universal speedups. No CPU/GPU speedup is assigned to Mini-Batch or ANN
variants because they perform different assignment work.

The summarized per-fit records and table source are preserved in
`outputs/20261026-real-quality-matrix-completion/`. The eight source cases are
the 1024/2048 quality folders named in that manifest. Every source archive has
a recursive `checksums-all.sha256`; the combined summary archive does too.
The combined archive contains `individual_runs.csv`, `method_summary.csv`,
`manifest.json` and `quality_target_hit_rate.png`. The plot groups the exact
C0/G0 lines and the five CUDA Mini-Batch/ANN/adaptive lines because their
aggregate hit counts match at every cell. It shows attainment counts, not
SSE convergence curves. The CSV retains every unique measured row, including
processed observations, distance-evaluation counts, quality-monitoring work
and internal timing categories. External `wall_s` is the primary complete-fit
measure; internal timing fields are reported separately and are not added
together. The summary does not pool different sessions.

#### Why the approximate/adaptive variants missed

The run metadata shows the majority of quality misses at this cap terminated
with `max_steps`, not tolerance convergence. The Mini-Batch update equations
and CPU/CUDA implementations passed the independent numerical and real-GPU
regression suites; final labels and SSE were recomputed exactly. There is no
identified implementation defect explaining these particular misses. The
large-crop cap gives Mini-Batch fewer than one full data pass, which is
insufficient evidence to conclude that the target is unreachable. Conversely,
the original cumulative-count fixed and adaptive Mini-Batch runs through 5,000
updates still missed, though their exact SSE continued to decline slowly. A
separate, matched validation diagnostic below tested a decayed-count update;
its result does not change the baseline or the frozen held-out policy. The
held-out 30STG K=16 study also missed the target for every approximate/adaptive
method at the frozen 1,000-update limit; it was not retuned.

The adaptive histories show that the frozen controller often had no trigger
to apply: no transitions were recorded in the 1.05M K=8/16/32 cases or the
4.19M K=4/8 cases. At 4.19M, deterioration/oscillation caused only isolated
batch/effort increases in some K=16/32 seeds. Thus adaptation benefit is not
established; it is a measured negative/neutral outcome for these frozen
settings, not proof of a controller implementation defect. GPU projection
ANN's separate negative feasibility result remains unchanged: approximate
candidate search was slower end-to-end than exact GPU search in the tested
synthetic range.

#### Held-out interpretation and remaining limits

The 30STG held-out quality report remains K=16 with its previously frozen
settings. The plan's K=4/8/16/32 quality sweep is covered on development and
validation imagery; additional K values were not used to tune on the held-out
scene. A supplemental held-out K=4 evaluation ran the CPU group to completion
(150 fits), then its CUDA group was intentionally interrupted after 14 of 150
fits. Both archives are preserved under `outputs/20261026-test-quality-k4-*`
and excluded from all conclusions; no policy or threshold was changed.

The implementation plan asks for kernel-level profiling support in M7 and
lists kernel/synchronization/transfer timing among desired measurements. This
was attempted, but Nsight Systems could not produce a usable WSL trace because
its importer was unavailable. Per-kernel bottleneck profiling is therefore a
documented limitation; grouped event timings are not presented as isolated
kernel timings. External labeled land-cover accuracy is optional in the plan
and unavailable because no aligned labeled reference raster was supplied.
Cluster IDs and the reported silhouette/DBI values are unsupervised metrics,
not semantic land-cover accuracy.

The prepared real-image N/K matrix uses one measurement session per case and
one or two measured fits per seed at large N, not five repeated sessions.
Independent-session replication across every N/K/method remains outstanding.
The untouched test scene has five seeds at K=16; it was not used to tune.

### Matched cumulative-versus-decayed Mini-Batch diagnostic — 26 October 2026

This single validation-scene diagnostic isolates statistic retention while
holding the training observations fixed. It uses the 31UFU validation crop
(N=262,144, D=4), K=16, seed 0, the same C0 and exact CPU assignment, a
pre-generated sequence of 5,000 batches of 8,192 indices drawn uniformly
without replacement within each batch, and identical exact full-data SSE
checkpoints every 100 updates. The cumulative-count replay reproduced every
full-data SSE checkpoint in the earlier fixed Mini-Batch archive, confirming
the data, initialization and sampling stream. The untouched 30STG test scene
was not used.

The experimental variant changes only the sufficient-statistic retention:
`S <- lambda*S + batch_sums`, `n_eff <- lambda*n_eff + batch_counts`, and
`centroid <- S/n_eff`, with
`lambda = 0.5 ** (batch_size / (10*N)) = 0.9978362593` per update. Thus past
batch contributions have a 10-full-data-pass half-life (2,621,440 processed
observations). This factor was used for this one diagnostic; it was not swept
or selected against the held-out scene.

| Update policy | Exact full SSE at update 2,000 | First checkpoint at/below frozen target | Exact full SSE at update 5,000 | Per-update centroid movement at 5,000 | Total wall time* |
|---|---:|---:|---:|---:|---:|
| Original cumulative counts | 29,353.42 | Not reached | 28,276.14 | 0.0000249 | 9.14 s |
| Experimental decayed counts | 25,657.99 | Update 2,100 (SSE 25,464.38) | 24,703.84 | 0.0001883 | 9.34 s |

The frozen target is 25,614.79 SSE (5% above the exact reference 24,395.03).
The decayed run first passed it at the recorded 2,100-update checkpoint; its
final SSE is 1.27% above the exact reference and therefore within that target.
The cumulative baseline did not reach it. At update 5,000, cumulative counts
sum to 40.96 million historical observations; decayed effective counts sum to
about 3.79 million. No clusters had zero counts in either run. The movement
and SSE differences support historical-count inertia as a material cause in
this K=16 validation case. They do not establish the best forgetting factor,
convergence to a global optimum, or generalization to other scenes, seeds or K.

*These are one diagnostic timing per method, after JIT warm-up, with matching
batch gather/assignment/update work and exact full-data checks included. Their
small difference is not evidence of a speedup.

The production cumulative-count implementations `C1`/`G1` remain unchanged
and remain the fixed Mini-Batch baselines. The decayed-count rule is a separate
experimental diagnostic, not a frozen-policy result and not part of the
held-out test evaluation. The project should report both the cumulative
baseline's target miss and this one validation-only positive result without
replacing the baseline or claiming the alternative works generally.

Artifacts and checksums are preserved under
`outputs/20261026-validation-decayed-count-k16-b8192-seed0/`: the exact runner
used, the portable reproduction script snapshot, configuration/provenance,
checkpoint CSV, the 164 MB pre-generated batch-index array, source snapshots,
and `checksums.sha256`. All listed artifact checksums were verified. Reproduce
to a new output directory (the command refuses to overwrite an existing one):

```bash
.venv/bin/python -B experiments/diagnostic_decayed_counts.py \
  --scene-dir outputs/20261025-independent-holdouts/validation512 \
  --output-dir outputs/decayed-count-validation-NEW \
  --reference-jsonl outputs/20261025-validation-minibatch-5k-k16-b8192-seed0/raw.jsonl \
  --k 16 --seed 0 --updates 5000 --batch-size 8192 \
  --checkpoint-interval 100 --forgetting-half-life-passes 10 \
  --target-sse 25614.78637147278
```

The scene array and reference JSONL are local ignored outputs and must exist at
those paths or be supplied at equivalent paths. The script verifies the
reference X/C0 hashes and, when a matching record is supplied, checks the
cumulative replay against its saved full-data SSE checkpoints. Its CLI/help and
syntax were checked; the portable copy was not executed end-to-end after
refactoring, to avoid repeating the experiment. The exact script that produced
the reported data is preserved separately and hashed in the archive manifest.
