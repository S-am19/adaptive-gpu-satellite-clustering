# Adaptive GPU-Accelerated Mini-Batch K-Means for Sentinel-2 Pixel Clustering

**Master manuscript — evidence-checked draft, 26 October 2026**

**Authors:** Samridhi [full name and affiliation to confirm] and Daksh [full name and affiliation to confirm]

**Course:** Parallel Computing

**Status:** Content manuscript only. Apply the official IEEE and Springer templates after the instructor confirms the author roster and acceptable CUDA language/toolchain. This file is not either final publisher layout.

## 1. Title

Adaptive GPU-Accelerated Mini-Batch K-Means for Sentinel-2 Pixel Clustering

The project title is retained. The original faculty-commented proposal requested a clearer title and an explicit expansion of ANN. The manuscript uses “approximate nearest-neighbor (ANN) centroid assignment” and does not use ANN to mean artificial neural network.

## 2. Abstract

Large multispectral satellite scenes contain millions of valid pixels, making repeated nearest-centroid assignment a computationally important part of K-Means clustering. This project implements a sequential CPU reference, custom Python-authored Numba CUDA kernels, fixed and adaptive Mini-Batch variants, CPU HNSW approximate nearest-neighbor assignment, and CUDA projection-based candidate assignment. A Rasterio-based Sentinel-2 Level-2A pipeline reads B02, B03, B04, and B08, applies product metadata calibration and Scene Classification Layer masking, standardizes with development-scene statistics, and preserves geospatial pixel mapping.

Experiments use genuine 10 m Sentinel-2 crops from one development acquisition and separate validation and held-out acquisitions. Across a validation K sweep at 262,144 pixels, paired exact CUDA-to-CPU median speed ratios (CPU time divided by GPU time) were 0.744, 0.930, 1.363, and 3.283 for K=4, 8, 16, and 32, respectively; the low-K cases were slower on CUDA. In the held-out K=16 study, exact CPU and CUDA each reached a frozen 5% relative-SSE target in 25/25 runs, whereas every tested approximate or adaptive method reached its 1,000-update cap without reaching the target. On large development crops, exact CUDA was faster in selected target-reaching paired cases, with observed ratios up to 5.624, but exact methods also missed the target in some high-K runs. A one-update fixed-work exact CUDA study measured median paired speed ratios of 1.087 at 1,048,548 pixels and 2.052 at 4,194,265 pixels; these are not convergence speedups. CPU HNSW and the tested CUDA projection ANN did not establish an end-to-end GPU ANN advantage. A one-seed validation diagnostic found lower SSE from a separately labeled decayed-count update, while an observation-budget Adaptive V2 comparison found similar quality to fixed decay and slower time-to-target than exact CUDA.

The evidence supports a workload-dependent exact CUDA advantage and several meaningful negative results, not universal acceleration. The crops have no aligned semantic ground truth, so cluster IDs are not validated land-cover classes. Broad multi-session scaling, real-image D=14 evaluation, systematic peak memory, and kernel-level profiling remain unavailable. Instructor acceptance of Python-authored CUDA under the proposal’s C/C++ wording remains pending.

## 3. Keywords

Parallel computing; K-Means; CUDA; Sentinel-2; satellite image clustering; Mini-Batch K-Means; approximate nearest-neighbor assignment; adaptive computation.

## 4. Introduction

Optical Earth-observation products provide spatially dense measurements that can be explored without labels using unsupervised clustering. In a pixel-level K-Means workload, each valid pixel is represented by a multispectral feature vector. For N pixels, D bands, and K centroids, a direct assignment pass evaluates O(NKD) feature operations. Repeating this pass during Lloyd iterations makes parallel nearest-centroid assignment attractive, while transfer, reduction, initialization, convergence checks, and final evaluation remain part of end-to-end cost.

This project studies those tradeoffs using Sentinel-2 Level-2A imagery. It treats exact sequential CPU K-Means as a numerical reference and compares it with a custom CUDA implementation. Mini-Batch variants reduce per-update observations but can introduce slower or less reliable convergence. ANN assignment attempts to reduce centroid comparisons but incurs index or candidate-generation overhead. Adaptive variants select batch size, ANN effort, or both from observed progress signals.

The question is empirical: under what image sizes and cluster counts does custom CUDA reduce complete-fit time at equal work or comparable quality, and do Mini-Batch, ANN, or adaptation improve end-to-end time to a fixed objective threshold? The project does not assume that an approximate method wins. This framing follows GPU K-Means work that treats data movement, batching, and distance work as coupled costs [3], and remote-sensing studies that evaluate clustering under scene-specific preprocessing and quality limitations [1], [2], [6], [7].

## 5. Literature review

Recent work spans parallel K-Means for satellite imagery, GPU-exact acceleration, approximate assignment, Mini-Batch convergence, and large-data decomposition. Pant et al. compare parallel variants for satellite-image K-Means, while Han and Lee study CPU/GPU parallelization for inter-image consistency across image sequences [1], [2]. Li et al. present a memory-aware batched GPU Lloyd variant that uses geometric bounds to avoid redundant assignment and transfer work [3]. These studies motivate separating algorithm compute from complete-fit costs and evaluating multiple scene workloads. He et al. provide complementary CPU–GPU evidence using a two-step summation method to limit centroid-update rounding error; the journal issue is dated 2022, though its online-first date is borderline for a strict five-year cutoff [5].

Mini-Batch K-Means trades full-dataset updates for lower per-iteration work, but centroid staleness can affect quality. Zhu et al. explicitly study staleness reduction, and Schwartzman analyzes conditions under which batch-level progress can support termination [4], [8]. Neither result implies that a fixed or probe-based stopping rule will satisfy this project’s scene-specific 5% objective target. For large-data clustering, Mussabayev et al. examine decomposition-based K-Means strategies [9]. These methods show that reduced data processing can be useful, while underscoring the need to compare at an equivalent objective or observation budget.

Remote-sensing applications also differ in temporal structure and label availability. Guo et al. jointly use iterative clustering in a temporal representation-learning system; Paradis studies probabilistic unsupervised classification on large spectral-imaging datasets [6], [7]. The present work instead clusters a single-date four-band reflectance vector and reports unlabeled clusters. It does not claim land-cover classification accuracy.

For large K, approximate nearest-neighbor search has been proposed to reduce the dominant assignment cost; Spalding-Jamieson et al. describe seeded approximate search graphs for large-k workloads [10]. That work is an arXiv preprint, not counted here as a peer-reviewed article. The present prototype uses CPU HNSW and a distinct CUDA projection shortlist, and its measured feasibility results are negative for the tested ranges. It is therefore not presented as an implementation or validation of the seeded-graph method.

The bounded gap studied here is an integrated, reproducible comparison on real Sentinel-2 crops that reports exact SSE, stopping reason, repeated seeds, and full-fit time while retaining negative Mini-Batch and ANN results. The evidence does not establish a new general clustering algorithm, a universal crossover, or semantic land-cover accuracy. Eight peer-reviewed references [1]–[4] and [6]–[9] fall within the rolling five-year window by their publication dates. Reference [5] is an additional CPU/GPU paper listed in the 2022 journal volume, but its online-first date is just outside a strict rolling cutoff; [10] is a recent preprint and is not counted as peer reviewed. An older satellite time-series study by Zhang et al. is retained as background [11].

## 6. Problem statement

Given valid reflectance vectors X={x_i} in R^D and a requested K, minimize the within-cluster sum of squared Euclidean distances:

\[
J(C)=\sum_{i=1}^{N}\min_{1\leq j\leq K}\|x_i-c_j\|_2^2,
\qquad C=(c_1,\ldots,c_K). \tag{1}
\]

The exact baseline evaluates every point-centroid distance. Mini-Batch methods update centroids from a subset of observations; ANN methods return a candidate nearest centroid rather than necessarily evaluating all K centers; adaptive methods change computation effort using measurable progress signals. The evaluation problem is to quantify the time/quality tradeoff without treating different observation counts as equivalent work or treating a capped run as converged.

## 7. Objectives and research questions

The project objectives are: (i) implement and validate sequential CPU and exact custom CUDA Lloyd K-Means; (ii) prepare genuine Sentinel-2 Level-2A data and reconstruct spatial cluster maps; (iii) evaluate fixed Mini-Batch, CPU ANN, CUDA approximate assignment, and adaptive ablations; (iv) preserve reproducible evidence, including negative findings.

The corresponding questions are:

1. How does paired exact CPU/CUDA complete-fit performance vary with N and K?
2. How often do approximate and adaptive methods reach a frozen objective gap under specified caps?
3. Does CPU HNSW or CUDA projection assignment reduce total assignment or fit cost after index/candidate maintenance?
4. Do adaptive batch or effort changes improve time-to-target compared with fixed settings?
5. What is demonstrated on separate development, validation, and held-out scenes, and what remains unsupported without labeled ground truth?

## 8. Proposed methodology

### 8.1 Sentinel-2 preprocessing and split policy

The inputs are genuine Sentinel-2 Level-2A SAFE products. The pipeline reads B02 (blue), B03 (green), B04 (red), and B08 (near-infrared) at 10 m and aligns the categorical 20 m SCL to the band grid with nearest-neighbor georeferenced resampling. Product XML quantification and per-band offsets are applied after no-data/saturation handling. This processing follows the Level-2A product structure and surface-reflectance/SCL description in the Copernicus product documentation [12]. The implemented exclusion set is SCL classes 0, 1, 3, 8, 9, and 10. Classes 2, 7, and 11 are retained by the current policy. The development crop contained no excluded cloud/shadow/no-data SCL pixels; retaining low-probability cloud class 7 and snow/ice class 11 is a limitation, not evidence that these classes are cloud-free or snow-free.

Development is Sentinel-2A tile 31UFS, acquired 25 June 2024. Validation is Sentinel-2B tile 31UFU, acquired 20 July 2024. Held-out test is Sentinel-2A tile 30STG, acquired 4 June 2024. Each reference 512×512 crop contains 262,144 selected valid pixels. Development standardization statistics were fitted on the Maastricht development crop and reused unchanged for validation and test. Larger development matrices use nested crops from that same acquisition; they are not independent scenes. All real-image inputs have D=4. D=14 experiments are synthetic only.

### 8.2 Objective and quality protocol

For each seed s, the evaluation reference is the final exact CPU SSE from the prescribed reference run. A result reaches the frozen threshold when

\[
\mathrm{SSE}_{m,s}\leq 1.05\,\mathrm{SSE}_{\mathrm{ref},s}, \quad
g_{m,s}=100\frac{\mathrm{SSE}_{m,s}-\mathrm{SSE}_{\mathrm{ref},s}}
 {\mathrm{SSE}_{\mathrm{ref},s}}\leq 5\%. \tag{2}
\]

This target is used only by the evaluation runner, never by the adaptive controllers. A target hit is distinct from tolerance convergence. Every final SSE and returned label assignment is an exact full-dataset evaluation at the returned centroids. Termination at `max_steps` or `observation_budget` is recorded as capped, not converged.

### 8.3 Timing and statistical treatment

The primary runtime is external complete-fit wall time. Exact CPU/GPU paired comparisons reuse the same X, initial centroids, seed, and fit configuration. Warm-ups are excluded from reported repeated measurements where the manifest says they were used. Median and interquartile range (IQR) summarize repeated timings. A speed ratio is paired CPU wall time divided by paired GPU wall time; ratios above one indicate GPU faster. CUDA device/event timing categories are diagnostic and can overlap host-side intervals; they are not added to derive total time. Index build/rebuild, sampling, monitoring, transfer and final exact evaluation are reported as separate measurements only where the record contains them.

## 9. System architecture

The data path is SAFE/XML/JP2 bands and SCL → calibrated and masked reflectance → development-fitted standardization → contiguous float32 feature matrix and pixel-index map → seeded centroid initialization → clustering backend → exact final labels/SSE → georeferenced label raster and visualization. The preprocessing module preserves crop transforms and valid-pixel positions so results can be reconstructed into the image grid.

The compute path has a shared Python contract and distinct implementations: `src/pcap/kmeans_cpu.py` is the sequential Numba-JIT Lloyd baseline; `src/pcap/kmeans_cuda.py` provides custom Numba CUDA kernels; `src/pcap/minibatch.py` and `minibatch_cuda.py` implement fixed Mini-Batch; `assignments_ann.py` is CPU HNSW; `assignments_ann_cuda.py` is a custom CUDA projection candidate search; `adaptive_minibatch.py`, `adaptive_cuda.py`, `adaptive_ann_controller.py`, and `adaptive_combined.py` implement batch, effort, and combined control. CUDA entry points require an available real CUDA device and fail explicitly rather than silently substituting CPU execution.

## 10. Algorithms

### 10.1 Exact Lloyd K-Means

At iteration t, assignment uses lowest-index tie-breaking:

\[
z_i^{(t)}=\arg\min_j\|x_i-c_j^{(t)}\|_2^2. \tag{3}
\]

For each nonempty cluster, the next centroid is its assigned-point mean:

\[
c_j^{(t+1)}=\frac{\sum_{i:z_i^{(t)}=j}x_i}{n_j^{(t)}}. \tag{4}
\]

Empty clusters retain their prior centroid. Float64 accumulations and squared-distance calculations reduce numerical error; public feature and centroid arrays remain float32. Initialization samples distinct row indices from X with a seeded NumPy generator; corresponding method comparisons share C0. Termination follows the configured displacement/probe criteria or an explicit step cap. Final labels and SSE are recomputed exactly after termination.

The CUDA implementation assigns points in parallel with custom Numba CUDA kernels and keeps X and working buffers resident on the device during iterative execution. GPU atomic cluster accumulation uses float64 sums and int64 counts; synchronization is used around timed kernel groups. The code is Python-authored CUDA, not CUDA C/C++ source. A CUDA C++ requirement interpretation therefore requires instructor confirmation.

### 10.2 Fixed Mini-Batch

Each update samples a uniform batch without replacement; separate batches may revisit points. The fixed cumulative update stores per-cluster running sums S_j and counts n_j across updates:

\[
S_j\leftarrow S_j+\sum_{x\in B:z(x)=j}x,\quad
n_j\leftarrow n_j+|\{x\in B:z(x)=j\}|,\quad
c_j\leftarrow S_j/n_j.
\]

These statistics begin at zero and do not count C0 as an observation. A cluster not observed in a batch keeps its centroid. Batch objectives are diagnostics and may increase; they are not the reported final full-data SSE. CPU and CUDA versions preserve persistent statistics and seeded sampling. The probe-based convergence heuristic is not a guarantee of full-data convergence.

### 10.3 ANN and ANN adaptation

CPU ANN uses HNSW over centroid vectors with deterministic seed, explicit graph parameters, single-threaded query, and explicit index refresh when centroids change. Query distances are recomputed against selected centroids; approximate labels can differ from exact. Search effort is an HNSW query-effort parameter. On GPU, the implemented approximate backend is projection-based candidate filtering, not HNSW; it maintains device-side projection structures and evaluates a candidate subset, with exact behavior used at or below the configured small-K threshold. Both index/candidate maintenance and final exact assignment are part of complete fit cost.

Adaptive ANN effort uses observable probe objective trends, normalized centroid movement, and probe-label changes. Its policy has descending effort levels, deterioration rollback/increase behavior and a quiet-checkpoint step-down rule. The validation conservative configuration used batch bounds 4,096–65,536, a 1% deterioration threshold, movement threshold 0.001, label-change threshold 0.005, relative objective-change threshold 0.001, and ten quiet checkpoints for batch reduction. ANN effort used levels 64/32/16/8 on CPU and 16/8/4/1 on CUDA, with three quiet checkpoints and 1% deterioration. These are fixed policy settings, not learned optimal values. Maximum-batch confirmation freezes adaptation; it is a stopping safeguard, not proof of global optimality.

### 10.4 Experimental decayed-count and Adaptive V2 variants

The original cumulative algorithm remains the baseline. A separate diagnostic decays historical statistics before adding each batch:

\[
S_j\leftarrow\lambda S_j+S_{j,B},\qquad
n_j^{\mathrm{eff}}\leftarrow\lambda n_j^{\mathrm{eff}}+n_{j,B},\qquad
c_j\leftarrow S_j/n_j^{\mathrm{eff}}. \tag{5}
\]

with observation-scaled \(\lambda=0.5^{b/(10N)}\), corresponding to a nominal ten data passes per history half-life. This is an experimental update policy, not the accepted fixed Mini-Batch baseline. Adaptive V2 adds a separate deterministic batch controller driven by observable probe statistics. Neither controller receives the frozen exact-reference threshold. The matched V2 experiment is small and unreplicated; it does not support a general efficiency claim.

## 11. Experimental setup

The primary verified environment is Ubuntu 24.04 in WSL2, Python 3.12.3, an NVIDIA GeForce RTX 4050 Laptop GPU with 6,141 MiB reported device memory, and NVIDIA driver 581.86. The validated Python stack recorded by experiment manifests includes NumPy 2.2.6, Numba 0.65.1, numba-cuda 0.30.4, hnswlib 0.8.0, Rasterio 1.5.1 (bundled GDAL 3.12.4), Matplotlib 3.11.2, and pandas 3.1.0. CUDA toolkit was recorded as 12.0. These versions describe the recorded environment; they are not all mandatory application requirements. WSL GPU discovery requires the WSL driver library path `/usr/lib/wsl/lib`, configured only by the project launcher.

| Scene role | Sentinel-2 product | Reference crop | Grid / features | Provenance |
|---|---|---|---|---|
| Development | S2A 31UFS, 2024-06-25 | Maastricht 512×512 window; row 6084, column 8679; larger nested development crops for selected studies | EPSG:32631, 10 m, B02/B03/B04/B08, D=4; N=262,144 in the reference crop | `outputs/20261024-real-scene-validation/`, `outputs/20261024-real-maastricht-512/`, per-study manifests |
| Validation | S2B 31UFU, 2024-07-20 | `validation512`, row 8192, column 5888, 512×512 | EPSG:32631, 10 m, D=4, N=262,144 | `outputs/20261025-validation-scene-k16-exact/render_G0/provenance.json` and study manifests |
| Held-out test | S2A 30STG, 2024-06-04 | `test512`, row 1280, column 6144, 512×512 | EPSG:32630, 10 m, D=4, N=262,144 | `outputs/20261025-test-scene-all-method-maps/render_G0/provenance.json` and quality archives |

All study-specific seeds, repeats, warm-ups, caps, policy settings, hashes, timing details, and session hardware snapshots are recorded in the corresponding ignored output manifests. The hardware data in this manuscript is a session-level record, not per-fit GPU telemetry.

## 12. Results

### 12.1 Validation time to the frozen quality target

At N=262,144 and D=4, exact C0 and G0 reached the 5% target in all 25 measured fits for each K. The paired timing ratios reveal a K-dependent crossover.

**Table 1. Paired exact CPU/GPU validation time-to-target results.** Source: `outputs/20261025-validation-quality-k-sweep-summary-v2/exact_speedup.csv`; five seeds × five measured fits, two warm-ups, max 100 updates. Ratio is median of paired C0/G0 wall-time ratios; IQR is the IQR of paired ratios.

| K | C0 median wall (s) | G0 median wall (s) | Paired CPU/GPU ratio (IQR) | Exact target hits, C0/G0 |
|---:|---:|---:|---:|---:|
| 4 | 0.1282 | 0.1825 | 0.744 (0.341) | 25/25 each |
| 8 | 0.3737 | 0.3832 | 0.930 (0.204) | 25/25 each |
| 16 | 0.7828 | 0.5506 | 1.363 (0.211) | 25/25 each |
| 32 | 1.3871 | 0.4180 | 3.283 (1.536) | 25/25 each |

The ratio definition makes the first two values evidence that CUDA was slower at K=4 and slightly slower at K=8 in this workload. The K=32 ratio has a broad IQR. The complete per-K summaries and manifests retain SSE, step counts, termination and pairing hashes.

### 12.2 Development target attainment at larger N

The development quality matrix held the 5% target fixed and capped runs at 100 steps. The table reports C0/G0 target hits and the matching per-method hit count for each of the five CUDA approximate Mini-Batch/ANN/adaptive variants (G1, G2CUDA, G3CUDA, G4CUDA, G5CUDA). Each variant’s hit count matched within the plotted cell; the denominator is not pooled across variants.

**Table 2. Development-scene target hits by N and K.** Source: `outputs/20261026-real-quality-matrix-completion/method_summary.csv`, `individual_runs.csv`, and `manifest.json`. N values are valid pixels from nested crops; measured repeats are 10 for N=1,048,548 at K=4/8/16, 5 for K=32, and 5 per cell at N=4,194,265.

| N | K | C0 hits | G0 hits | Each approximate CUDA variant hits | Paired exact CPU/GPU ratio where all exact pairs reached target |
|---:|---:|---:|---:|---:|---:|
| 1,048,548 | 4 | 10/10 | 10/10 | 10/10 | 1.387× (IQR 0.392×) |
| 1,048,548 | 8 | 10/10 | 10/10 | 2/10 | 2.484× (IQR 0.997×) |
| 1,048,548 | 16 | 6/10 | 6/10 | 0/10 | Not summarized as a complete common-target cell |
| 1,048,548 | 32 | 4/5 | 4/5 | 0/5 | Not summarized as a complete common-target cell |
| 4,194,265 | 4 | 5/5 | 5/5 | 3/5 | 3.968× (IQR 0.699×) |
| 4,194,265 | 8 | 5/5 | 5/5 | 0/5 | 5.624× (IQR 0.859×) |
| 4,194,265 | 16 | 3/5 | 3/5 | 0/5 | Not summarized as a complete common-target cell |
| 4,194,265 | 32 | 3/5 | 3/5 | 0/5 | Not summarized as a complete common-target cell |

For K=16 and K=32, exact methods also missed the cap-dependent target in some runs. Most approximate/adaptive outcomes ended at `max_steps`. The work at N=4.19M and a 100-step limit is roughly 0.20 passes for batch size 8,192; this is not enough to interpret every miss as an algorithmic impossibility. The target-hit counts are the recorded evidence, not a claim of convergence.

### 12.3 Held-out test scene

The held-out scene study evaluates K=16, N=262,144, D=4, seeds 0–4, five measured runs per method and two warm-ups. The exact CPU and exact CUDA groups each reached the threshold in 25/25 runs. Every tested Mini-Batch, HNSW, batch-adaptive, ANN-effort-adaptive, and combined variant reached the 1,000-step cap and recorded zero target hits. CPU and CUDA were measured in separate execution groups; their wall-time medians must not be divided into a speedup.

**Table 3. Held-out target and representative result summary.** Source: `outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/aggregate_method_summary.csv` and its manifest. “Median SSE” is full-dataset exact SSE. Wall medians are within-group only.

| Group | Method | Target hits / 25 | Median final SSE | Median wall (s) | Stop status |
|---|---|---:|---:|---:|---|
| CPU | C0 exact | 25 | 32,615.65 | 5.510 | target reached |
| CUDA | G0 exact | 25 | 32,615.65 | 1.442 | target reached |
| CPU | C1 fixed Mini-Batch | 0 | 39,672.47 | 1.496 | max_steps=1,000 |
| CUDA | G1 fixed Mini-Batch | 0 | 39,672.47 | 1.300 | max_steps=1,000 |
| CPU | G2 HNSW fixed; G3 batch-adaptive; G4 HNSW-effort adaptive; G5 combined | 0 each | 39,672–39,711 | 1.368–9.805 | max_steps=1,000 |
| CUDA | G2CUDA projection; G3CUDA batch-adaptive; G4CUDA effort-adaptive; G5CUDA combined | 0 each | 39,647–41,187 | 5.746–11.332 | max_steps=1,000 |

The separate map-generation quality sample (seed 0, deterministic 3,000-pixel sample) reports C0/G0 Silhouette 0.3396 and Davies–Bouldin index 0.9018. For C1/G1, the corresponding values are 0.3208 and 0.9589. These are sample metrics for one map-generation fit, not repeated estimates and not semantic accuracy. The quality CSV and manifest are in `outputs/20261025-test-scene-all-method-maps/`.

**Table 4. Sampled internal clustering metrics for map-generation fits.** Source: `outputs/20261025-test-scene-all-method-maps/quality_metrics_sampled.csv` and `quality_metrics_manifest.json`.

| Method | Exact full-data SSE | Silhouette (sample 3,000) | Davies–Bouldin (sample 3,000) | Recorded stopping |
|---|---:|---:|---:|---|
| C0 / G0 exact | 32,485.55 | 0.3396 | 0.9018 | tolerance; convergence satisfied |
| C1 / G1 fixed Mini-Batch | 40,449.44 | 0.3208 | 0.9589 | max_steps=1,000; not converged |

These rows are from one map-generation seed-0 record per method, not the repeated frozen target matrix. Higher silhouette and lower Davies–Bouldin are preferable as internal cohesion/separation summaries, but they are not accuracy against land-cover truth.

### 12.4 Exact fixed-work scaling

The real development fixed-work study used D=4, K=16, seeds 0–4, two warm-ups, five measured repetitions, and exactly one Lloyd update. Paired C0/G0 comparisons produced a median ratio of 1.087 (IQR 0.090) at N=1,048,548 and 2.052 (IQR 0.696) at N=4,194,265. External paired median times were 0.1487/0.1369 s and 0.6302/0.2884 s for CPU/GPU. Both methods performed one update. These are single-session measurements on nested crops and measure one-update fixed work, not time-to-convergence.

**Table 5. Exact fixed-work complete-fit timing.** Source: `outputs/20261025-real-fixed-work-scaling-summary-v2/{exact_speedup.csv,method_summary.csv}`. Speedup is the median of matched CPU/GPU ratios; IQR is across paired ratios.

| N | D | K | Updates | CPU median (s) | GPU median (s) | Paired speedup (IQR) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,048,548 | 4 | 16 | 1 | 0.1487 | 0.1369 | 1.087 (0.090) |
| 4,194,265 | 4 | 16 | 1 | 0.6302 | 0.2884 | 2.052 (0.696) |

### 12.5 ANN, adaptation, and decayed-count diagnostics

CPU HNSW and the CUDA projection candidate method did not show a general advantage. In a synthetic feasibility study (N=8,192, D=4/14, K=4–256, effort 1/4/16), approximate CUDA modes had greater median complete cost than exact GPU search in the tested cases; low-effort recall ranged from 0.8% to 16.6%. K≤8 used exact behavior under the configured threshold. This result is design- and range-specific. Real held-out HNSW and CUDA projection variants missed the frozen target at K=16 under the 1,000-update cap.

**Table 6. CUDA projection ANN feasibility boundary.** Source: `outputs/20261025-projection-feasibility-final/{raw.csv,summary.csv,manifest.json}`. The source records support the aggregate range below; individual configuration rows should be shown from the raw CSV if a more detailed figure is added.

| Evidence item | Recorded result |
|---|---|
| Synthetic input matrix | N=8,192; D=4 and 14; K=4, 8, 16, 32, 64, 128, 256; seed 0 |
| Search effort | 1, 4, 16; two warm-ups and five measured repetitions |
| Small-K behavior | K≤8 uses the configured exact path, so these are not approximate-search results |
| Low-effort assignment recall | 0.8%–16.6% across the tested D/K cases |
| Complete-cost comparison | Tested approximate CUDA modes had higher median complete cost than exact GPU assignment in the tested approximate modes |
| Scope | Synthetic feasibility study; not evidence for a real-image GPU ANN speedup |

The matched validation cumulative-versus-decayed diagnostic used 31UFU, K=16, seed 0, batch 8,192, 5,000 updates, the same initialization and pre-generated batch indices, and decay half-life ten data passes. Cumulative counts ended at exact SSE 28,276.14 without reaching the target. Decayed counts ended at 24,703.84 and first crossed the target at checkpoint 2,100. One run per policy took approximately 9.14 and 9.34 seconds, respectively. This suggests cumulative-history inertia in that case but does not estimate a general effect or speedup.

The Adaptive V2 observation-budget ablation used the validation crop, K=16, seeds 0/1/2, a common 40.96-million processed-observation budget, and shared 8.192-million observation checkpoints. The frozen target was evaluation-only. Fixed decay and V2 had similar final SSE; both missed seed 1. V2’s single-run per-seed elapsed times to the cap were 13.076/12.773/9.330 s, versus 26.580/31.012/23.459 s for fixed decay, with differing update schedules and sampled-index streams after batch widths diverged. Exact CUDA reached the target in 1.085–1.287 s. These one-run data suggest a throughput tradeoff for V2 against fixed decayed updates, but no time-to-target advantage over exact CUDA and no replicated adaptive efficiency advantage. The original summary writer failed; the labeled reconstructed CSV was derived from preserved raw records and is accompanied by the failure and recovery records.

**Table 7. Matched validation decayed-count and observation-budget diagnostics.** Sources: `outputs/20261026-validation-decayed-count-k16-b8192-seed0/checkpoint_metrics.csv` and `outputs/20261026-adaptive-v2-observation-budget-40m/measurements_reconstructed.csv`; the latter is reconstructed from preserved raw records and is not the missing original CSV.

| Diagnostic | Seed | Updates / observation cap | Final exact SSE | First crossing (checkpoint or elapsed time) | End-to-end time | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Cumulative counts | 0 | 5,000 updates | 28,276.14 | Not reached | 9.14 s | Matched replay baseline |
| Decayed counts, half-life 10 passes | 0 | 5,000 updates | 24,703.84 | Checkpoint 2,100 | 9.34 s | One paired seed; no timing advantage established |
| Exact CUDA G0 | 0 | Convergence run | 24,395.03 | 1.287 s | 1.443 s total | Reached frozen target |
| Fixed decay | 0/1/2 | 40.96M observations each | 24,703.84 / 24,167.06 / 25,429.70 | 13.702 s / not reached / 23.441 s | 26.580 / 31.012 / 23.459 s | One run per seed |
| Adaptive V2 decay | 0/1/2 | 40.96M observations each | 24,703.78 / 24,167.57 / 25,435.95 | 10.054 s / not reached / 9.306 s | 13.076 / 12.773 / 9.330 s | One run per seed; sampling diverged after batch schedules differed |
| Exact CUDA G0 | 1/2 | Convergence run | 22,705.77 / 24,446.55 | 1.208 / 1.085 s | 1.422 / 1.474 s total | Both reached frozen target |

The first target crossing is the first recorded checkpoint at or below threshold; end-to-end time is full run time including final exact assignment. The diagnostic and ablation are different protocols and should not be combined into one estimate.

## 13. Performance analysis

The results show a workload-dependent exact CUDA crossover. At the validation crop, CUDA was slower at K=4 and K=8 and faster at K=16 and K=32. The large development paired ratios were higher in selected K=4/8 cells, but those estimates come from one session, low repetitions, and nested windows from a single acquisition. The fixed-work study isolates assignment/update work at one iteration and should not be merged with target-terminated measurements.

The approximate methods reduce work per selected batch or search subset, but end-to-end runtime includes the work needed to reach comparable SSE, monitoring, index maintenance, data transfer, and exact final evaluation. Because the approximate variants failed to reach the held-out target within the cap, their shorter or longer cap times are not matched-quality speedups. GPU projection ANN’s synthetic negative result further shows that approximate candidate search can lose to a parallel exact kernel when K and D are modest. The CPU HNSW backend pays index construction and centroid-refresh costs; those costs are included in its complete-run evaluation.

Instrumentation is incomplete for some historical studies. Internal timing intervals may overlap. Session snapshots do not estimate peak memory or per-fit utilization, and the Nsight attempt did not produce a usable kernel timeline. No causal explanation of timing outliers or low GPU utilization is asserted.

## 14. Comparison with existing methods

Published work establishes that GPU assignment, memory-aware batching, Mini-Batch methods, and remote-sensing clustering are active research areas [1]–[9]. This project’s exact CUDA implementation follows the data-parallel assignment strategy, while the observed complete-fit speed depends on K and N. Unlike ASB K-Means [3], this implementation does not use triangle-inequality bounds to skip stable assignments. Its custom CUDA source is authored in Python via Numba-CUDA; the comparison with CPU is on the same data and starts, and includes full-fit costs under the stated timing protocol.

The cumulative Mini-Batch baseline differs from Lloyd’s full-dataset update because it updates means from repeated, independently sampled minibatches and retains historical observation counts. Adaptive methods use local probe signals; a small probe does not certify global convergence. The HNSW CPU adapter rebuilds against moving centroids, while the CUDA projection search has a different index/candidate structure. These methods are not directly interchangeable. The negative outcomes are therefore reported as implementation-specific feasibility findings, not as a claim that ANN or adaptation is intrinsically ineffective.

Prior satellite clustering work sometimes uses aligned labels to evaluate accuracy [6], [7]. This project has no aligned reference labels. Accordingly, it reports objective, internal clustering metrics, exact assignment agreement, and visual maps only. It does not call clusters “land-cover classes” or claim land-cover accuracy.

## 15. Conclusion

The project delivers a real-data pipeline and a tested family of sequential CPU, CUDA, Mini-Batch, ANN, and adaptive implementations. The strongest positive empirical finding is that exact custom CUDA can reduce complete-fit time on some larger or higher-K Sentinel-2 workloads while being slower at smaller K. The validation K sweep provides paired repeated evidence; selected large development cells show higher ratios but require replication before broad claims.

The principal negative finding is that tested Mini-Batch, ANN, and adaptive variants did not reach the frozen 5% objective threshold on the held-out K=16 scene within 1,000 updates. CUDA projection ANN also lost to exact GPU search in its tested synthetic feasibility range. A decayed-count diagnostic improved SSE in one matched validation run; the Adaptive V2 budget ablation suggests greater update throughput than fixed decay at similar final objective in a small unreplicated comparison, yet exact CUDA reached the same target far sooner. These are useful bounded results, not evidence of a general adaptive advantage.

No semantic land-cover accuracy, robust multi-session large-N scaling, real D=14 performance, systematic peak memory, or kernel-level bottleneck attribution is established. Final publication claims should preserve these boundaries and include the archived manifests and raw records with the report package.

## 16. Limitations, instructor decisions, and future work

The following are explicit decisions or limitations for final submission:

1. **CUDA language/toolchain:** the original proposal uses C/C++ language wording. Exact GPU kernels here are authored in Python with Numba-CUDA. Obtain instructor confirmation that Python-authored Numba CUDA satisfies the project requirement; if not, scope a C++ CUDA port as new work.
2. **Official authorship:** the annotated proposal’s submitted-by field is blank. Confirm the exact team roster, author names, affiliations, and contribution statements; do not infer authors from filenames or machine accounts.
3. **Ground truth:** no aligned land-cover label raster is available. Semantic accuracy, class names, and map correctness against land-cover truth cannot be claimed.
4. **Scenes and generalization:** three acquisitions were used, with K=16 held-out evaluation and D=4 only on real imagery. Larger development crops are nested samples from one tile. The evidence does not establish generalization across sensors, seasons, regions, K, or D.
5. **Approximate/adaptive quality:** target misses at fixed caps are real negative outcomes, but capped runs are not converged outcomes. V2 timing is unreplicated and its sample sequence diverges from fixed decay after batch sizes diverge.
6. **Profiling and resources:** no usable Nsight trace, per-kernel timing, systematic peak VRAM/RSS, or continuous GPU utilization is available.
7. **Template and submission format:** official IEEE/Springer templates were not supplied in this task. The guideline points to publication templates; this Markdown master must be transferred to the approved templates after confirmation. Presentation/demo screenshots, contribution statement, and any course portal submission evidence remain separate deliverables.

The next work is editorial and administrative: obtain those instructor decisions, finalize the 8–10-paper survey, transfer this content to the approved templates, create the required presentation/demo materials from genuine saved outputs, and ensure the ignored SAFE data and experiment archives are backed up. No further algorithm or experiment cycle is implied by this manuscript.

## 17. References

[1] Y. R. Pant, L. Leigh, and J. Fajardo Rueda, “Improving K-Means Clustering: A Comparative Study of Parallelized Version of Modified K-Means Algorithm for Clustering of Satellite Images,” *Algorithms*, vol. 18, no. 8, Art. no. 532, 2025, doi: 10.3390/a18080532.

[2] S. Han and J. Lee, “Parallelized Inter-Image k-Means Clustering Algorithm for Unsupervised Classification of Series of Satellite Images,” *Remote Sensing*, vol. 16, no. 1, Art. no. 102, 2024, doi: 10.3390/rs16010102.

[3] M. Li, E. Frank, and B. Pfahringer, “Large scale K-means clustering using GPUs,” *Data Mining and Knowledge Discovery*, vol. 37, pp. 67–109, 2023, doi: 10.1007/s10618-022-00869-6.

[4] X. Zhu, J. Sun, Z. He, J. Jiang, and Z. Wang, “Staleness-Reduction Mini-Batch K-Means,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 10, pp. 14424–14436, 2024, doi: 10.1109/TNNLS.2023.3279122.

[5] G. He, S. Vialle, and M. Baboulin, “Parallel and accurate k-means algorithm on CPU-GPU architectures for spectral clustering,” *Concurrency and Computation: Practice and Experience*, vol. 34, no. 14, Art. no. e6621, 2022, doi: 10.1002/cpe.6621.

[6] W. Guo, W. Zhang, Z. Zhang, P. Tang, and S. Gao, “Deep Temporal Iterative Clustering for Satellite Image Time Series Land Cover Analysis,” *Remote Sensing*, vol. 14, no. 15, Art. no. 3635, 2022, doi: 10.3390/rs14153635.

[7] E. Paradis, “Probabilistic unsupervised classification for large-scale analysis of spectral imaging data,” *International Journal of Applied Earth Observation and Geoinformation*, vol. 107, Art. no. 102675, 2022, doi: 10.1016/j.jag.2022.102675.

[8] G. Schwartzman, “Mini-batch k-means terminates within O(d/ε) iterations,” in *Proc. 11th International Conference on Learning Representations (ICLR)*, 2023. [Online]. Available: https://openreview.net/forum?id=jREF4bkfi_S.

[9] R. Mussabayev, N. Mladenovic, B. Jarboui, and R. Mussabayev, “How to Use K-means for Big Data Clustering?,” *Pattern Recognition*, vol. 137, Art. no. 109269, 2023, doi: 10.1016/j.patcog.2022.109269.

[10] J. Spalding-Jamieson, E. W. Robson, and D. W. Zheng, “Scalable k-Means Clustering for Large k via Seeded Approximate Nearest-Neighbor Search,” arXiv:2502.06163, 2025. **Preprint; not counted as peer-reviewed.**

[11] Z. Zhang, P. Tang, W. Zhang, and L. Tang, “Satellite Image Time Series Clustering via Time Adaptive Optimal Transport,” *Remote Sensing*, vol. 13, no. 19, Art. no. 3993, 2021, doi: 10.3390/rs13193993. **Supplementary background; outside the rolling five-year count as of 26 October 2026.**

[12] European Space Agency, “S2 Products,” *Copernicus Sentinel-2 SentiWiki*. [Online]. Available: https://sentiwiki.copernicus.eu/web/s2-products. Accessed: Oct. 26, 2026.

### Figure and table source index for report assembly

The eight figure assets are byte-identical copies in `outputs/report-assets-20261026/figures/`; hashes and full source paths are in `outputs/report-assets-20261026/asset_manifest.json`.

| Manuscript item | Asset or evidence source | Main sections |
|---|---|---|
| Fig. 1 development RGB/NIR/G0 composite | `fig01_development_scene_composite.png` | 8, 9, 12 |
| Figs. 2–3 held-out true color and G0 cluster map | `fig02_test_scene_true_color.png`, `fig03_test_scene_g0_clusters.png` | 8, 12, 15 |
| Fig. 4 development target hit rates | `fig04_development_target_hit_rate.png` | 12, 13 |
| Fig. 5 validation exact speedup by K | `fig05_validation_exact_speedup_by_k.png` | 12, 13 |
| Fig. 6 held-out NIR false-color view | `fig06_test_scene_false_color_nir.png` | 8, 12 |
| Fig. 7/8 held-out seed-0 CPU/CUDA SSE trajectories | `fig07_heldout_cpu_seed0_trajectory.png`, `fig08_heldout_cuda_seed0_trajectory.png` | 12, 13 |
| Table 1 validation paired speed and target | `exact_speedup.csv`, validation `method_summary.csv` | 11–14 |
| Table 2 development N/K target hits | `individual_runs.csv`, `method_summary.csv` | 11–14 |
| Table 3 held-out quality/stopping | `aggregate_method_summary.csv`, `seed_method_summary.csv` | 11–15 |
| Tables 4–7 internal metrics, fixed-work, ANN, and adaptive diagnostics | study summary CSVs and the referenced diagnostic report/manifest paths | 11–14 |

### Mapping to the 17 required report sections

| Required section | Manuscript content | Figures / tables | References |
|---|---|---|---|
| 1. Title | Title and author placeholders | — | — |
| 2. Abstract | Aim, implementation, scenes, headline results, limitations | Tables 1–7 summarized | — |
| 3. Keywords | Project terms | — | — |
| 4. Introduction | Satellite workload and motivation | Fig. 1 | [1]–[3], [6], [7], [12] |
| 5. Literature Review | Recent GPU, Mini-Batch, remote-sensing and ANN work | — | [1]–[11] |
| 6. Problem Statement | SSE objective and computation tradeoff | Equation (1) | [3], [8], [9] |
| 7. Objectives | Five measured research questions | Tables 1–7 as evidence | — |
| 8. Proposed Methodology | Data splits, target, evaluation and timing rules | Figs. 1–3; Tables 1–3 | [1], [2], [6], [7], [12] |
| 9. System Architecture | Preprocess-to-map flow and implementation modules | Figs. 1–3 | [3], [5] |
| 10. Algorithms | Lloyd, Mini-Batch, ANN, controllers, experimental decay | Equations (3)–(5) | [3], [4], [8]–[10] |
| 11. Experimental Setup | Hardware, scenes, packages, protocols and provenance | Tables 1–7 | — |
| 12. Results | Real maps, exact CPU/CUDA, Mini-Batch, ANN, and adaptive outcomes | Figs. 1–8; Tables 1–7 | — |
| 13. Performance Analysis | Paired ratios, fixed work, variability, overhead boundaries | Figs. 4–8; Tables 1, 2, 5–7 | [3], [5], [9] |
| 14. Comparison with Existing Methods | Scope-aware comparison and methodological differences | Tables 1–7 | [1]–[11] |
| 15. Conclusion | Findings with capped-run and generalization caveats | Figs. 4–8; Tables 1–7 | — |
| 16. Future Scope | Limits, instructor decisions, future verification | — | — |
| 17. References | IEEE-style bibliography; recency and preprint note | — | [1]–[12] |

The content aligns to all 17 report sections specified by the course template. Publisher-specific page layout, image placement rules, title-page fields, and reference styling remain to be applied after the official template and instructor decisions are confirmed.
