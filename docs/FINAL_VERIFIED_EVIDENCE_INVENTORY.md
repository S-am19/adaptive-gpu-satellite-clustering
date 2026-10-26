# Final verified evidence inventory and report plan

**Purpose:** identify which existing records support the research paper and
which claims are not supported. This is an evidence index, not the paper and
not a new experiment. It does not alter archived measurements. It was prepared
from the saved local experiment records and reports on 26 October 2026.

**Code revision reviewed:** `experiment-adaptive-v2` at `25f8f464` (Adaptive V2
budget ablation). Experiment archives are local, Git-ignored artifacts; keep
them with the raw Sentinel-2 inputs when backing up the project. Paths below
are relative to the repository root. Tables in this document quote already
reported summaries; no missing metric has been calculated here.

## 1. Dataset provenance and method coverage

All three genuine scene groups use Sentinel-2 Level-2A products and the native
10 m bands B02, B03, B04 and B08 (`D=4`). Development-derived feature
standardization is reused for validation and test. The real-image experiments
do not provide real `D=14` data; that dimensionality was evaluated with
synthetic inputs only.

| Split / role | Product and date | Evidence and method coverage |
|---|---|---|
| Development | `S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE`, 2024-06-25, tile 31UFS (Maastricht). Raw product: `data/raw/20240625-31UFS/`. | The 30-step real matrix at `outputs/20261024-real-evaluation/` has 360 measured records for `C0,C1,G0,G1,G2,G3,G3CUDA,G4,G5`, `N=65,536/262,144`, `K=4/16`, seeds 0/1 and five repeats. Later N/K studies include all twelve C/G methods in some cells; the 26 October combined quality summary explicitly covers `C0,G0,G1,G2CUDA,G3CUDA,G4CUDA,G5CUDA` at `N=1,048,548/4,194,265`, `K=4/8/16/32`, seeds 0–4, with one or two measured repetitions per seed depending on cell. The 1024-window `K=16/32` archives also include `C1,G2,G3,G4,G5` and their CUDA counterparts. Coverage is archive/cell-specific; do not imply every method used the same N/K/repetition matrix. |
| Validation | `S2B_MSIL2A_20240720T103629_N0510_R008_T31UFU_20240720T121155.SAFE`, 2024-07-20, tile 31UFU. Raw product: `data/raw/independent-scenes/validation/`. | The frozen-policy four-K sweep at `outputs/20261025-validation-quality-k{4,8,16,32}-frozen-policy/` evaluates all twelve methods (`C0,C1,G0,G1,G2,G2CUDA,G3,G3CUDA,G4,G4CUDA,G5,G5CUDA`) on `N=262,144`, `D=4`, five seeds and five measured fits per method/seed, with two warm-ups and a 100-update cap. This scene also supports the cumulative-vs-decayed diagnostic and Adaptive V2 observation-budget ablation; those are separate protocols, described below. |
| Held-out test | `S2A_MSIL2A_20240604T110621_N0510_R137_T30STG_20240604T171851.SAFE`, 2024-06-04, tile 30STG. Raw product: `data/raw/independent-scenes/test/`. | Frozen `K=16` five-seed study: all twelve methods, seeds 0–4, five measured fits per method/seed and two warm-ups; CPU and CUDA measurements were run in separate execution groups. The scene was not used to select settings. Exact CPU/GPU baselines reached the frozen target in 25/25 runs; every Mini-Batch/ANN/adaptive method reached the 1,000-step cap and missed it. Do not divide the separately grouped CPU/GPU medians into a speedup. Adaptive V2 evaluation is not recorded on this held-out scene. |

Method labels in the reported scope: `C0` exact sequential CPU; `G0` exact custom CUDA; `C1/G1` fixed CPU/CUDA Mini-Batch;
`G2/G4/G5` CPU HNSW fixed/adaptive-effort/combined variants;
`G3/G3CUDA` adaptive batch CPU/CUDA; and `G2CUDA/G4CUDA/G5CUDA` custom CUDA
projection-assignment fixed/adaptive-effort/combined variants. CUDA projection
ANN is not GPU HNSW and did not demonstrate an end-to-end advantage.

### Input image and map locations

The genuine source imagery is the SAFE product data above: B02/B03/B04/B08
JP2s in each product's `GRANULE/.../IMG_DATA/R10m/`, with the corresponding
SCL JP2 in `R20m/` and product calibration in `MTD_MSIL2A.xml`. The input
assets, checksums and scene/crop metadata are described in
`data/independent_scene_candidates.json`, `data/scene_candidates.json`, and
the experiment manifests. Raw data is not tracked by Git.

Existing cluster maps produced from actual method outputs:

- **Main real-scene visual:** `outputs/20261024-real-evaluation/real-scene-comparison.png`.
- **Held-out scene, exact reference:** `outputs/20261025-test-scene-k16-exact-cap1000/render_C0/` and `render_G0/`.
- **Held-out all-method map set:** `outputs/20261025-test-scene-all-method-maps/render_<METHOD>/`, with method-specific `clusters.png`, georeferenced `clusters.tif`, `true_color.png`, `false_color_nir.png`, provenance and checksums. This is a map/execution artifact, not a repeated quality study.
- **Validation exact reference maps:** `outputs/20261025-validation-scene-k16-exact/render_C0/` and `render_G0/`.
- **Early real smoke map set:** `outputs/20261024-real-smoke-512/maps-G0/`, `maps-G3CUDA/`, and `maps-G5/`.

For a paper map panel, use a true-color or false-color view and an aligned
cluster map from the *same crop*, with split, product, crop, K, method, mask
and display stretch in the caption. Cluster IDs are arbitrary unsupervised
labels; no map establishes semantic land-cover accuracy. Prefer a held-out
exact C0/G0 pair for a method-neutral visual example and show other method
maps only when the comparison is scientifically relevant. Verify the
GeoTIFF CRS/transform and archive checksum before copying figures into a
submission package.

## 2. Metric audit

| Metric | Existing evidence and where measured | Missing or limiting evidence |
|---|---|---|
| Exact full-data SSE / objective gap | `final_exact_sse` is present in real evaluation CSVs and final model records. Quality studies retain frozen target, target-hit status, final exact SSE, steps, termination reason and convergence status. Adaptive V2 records exact final SSE and first target crossing in `outputs/20261026-adaptive-v2-observation-budget-40m/measurements_reconstructed.csv`; cumulative/decayed checkpoints are in `outputs/20261026-validation-decayed-count-k16-b8192-seed0/checkpoint_metrics.csv`. | This is an objective measure, not a semantic accuracy label or proof of a global optimum. Keep target attainment separate from tolerance convergence and from reaching `max_steps`. |
| Silhouette and Davies–Bouldin | Early genuine-data repeated matrix `outputs/20261024-real-evaluation/measurements.csv` has `silhouette_score` and `davies_bouldin_index` for its 360 rows. The held-out all-method map archive has `quality_metrics_sampled.csv` with sample-3000 Silhouette/DBI and sample metadata. | They are not present in the later large-N quality matrix summary. The sample-3000 held-out metric is one seed/map protocol, not a full repeated all-scene matrix. No externally labeled land-cover accuracy exists. |
| Execution time | External full-fit wall time exists in the benchmark CSVs, including final exact evaluation within the documented fit boundary. Some records also expose compute, setup, sampling, assignment, update, monitoring, controller, quality-evaluation, final-assignment and index-maintenance categories. | Internal categories can overlap; do not sum them to recreate wall time. CPU and CUDA held-out test groups were separate and cannot be used for a paired speedup. Some old smoke timings are single fits. |
| CPU/GPU speedup | Paired exact C0/G0 summaries validate matching X/C0 and compatible work. Validation K-sweep speedups are in `outputs/20261025-validation-quality-k-sweep-summary-v2/exact_speedup.csv`. Fixed-work large-N exact speedups are in `outputs/20261025-real-fixed-work-scaling-summary-v2/`. Development quality target ratios are in `outputs/20261026-real-quality-matrix-completion/method_summary.csv` and the validated summary manifest. | Report only the pair/protocol actually checked. Do not calculate a CPU/GPU “speedup” for Mini-Batch or ANN methods with different work, or for separately executed CPU/GPU sessions. No universal speedup claim is supported. |
| Memory | Manifests preserve session-level available/total GPU memory, some RAM snapshots and selected device-buffer estimates (notably the Adaptive V2 manifest). | Systematic peak allocated VRAM and peak host RSS per fit are not recorded across the matrix. Do not call free-memory snapshots “peak memory use.” |
| GPU utilization / clocks / temperature | Some manifests record before/after or session-level GPU conditions (P-state, clock, temperature, utilization). | Not continuous per-fit telemetry; insufficient for causal explanations of timing outliers or claims of sustained utilization. |
| Distance work / assignment agreement | Exact/projection synthetic ANN CSVs record candidate count, distance evaluations, recall, distance agreement and objective gap. Real fixed-work summaries include centroid-distance evaluations and processed observations for some methods. | A common, directly comparable evaluation-count field is not populated for every real experiment. Do not infer fewer evaluations from an ANN label or effort setting alone. |
| Target hit rate / stopping | Validation, development and test summaries contain target hits, exact SSE, iterations, termination reasons and convergence flags. | Large-N approximate/adaptive miss rates are genuine outcomes under the stated caps; they are not evidence that those methods converged. The held-out K sweep is K=16 only. |
| Scalability | Real D=4 one-update exact scaling at `N=1,048,548` and `4,194,265`, `K=16`, seeds 0–4, two warm-ups and five repeats is summarized in `outputs/20261025-real-fixed-work-scaling-summary-v2/`. Development quality matrix spans `N=262,144`, `1,048,548`, and `4,194,265` across K cells; validation spans K=4/8/16/32 at N=262,144. | Large-N quality cells use one or two measured fits per seed in one session. Real D=14 and multi-session replication of every large-N/method cell are absent. Do not conflate one-update scaling with time-to-convergence. |

## 3. Verified experiment shortlist for the paper

Include the following source records, keeping each protocol distinct. Every
reported number in the paper should point to its raw/summary file and manifest;
preserve failed, partial and capped rows.

| Proposed table / plot | Source records and recorded protocol | Use and caveat |
|---|---|---|
| Real-scene input and map panel | `outputs/20261024-real-evaluation/real-scene-comparison.png`; exact C0/G0 aligned maps under validation/test paths above; all-method held-out maps under `outputs/20261025-test-scene-all-method-maps/`. Product manifests and `data/independent_scene_candidates.json` identify product/crop provenance. | Use one representative scene panel plus an aligned cluster output. Do not interpret cluster IDs as ground-truth classes. The older main composite comes from the development product, not the independent test product. |
| Validation quality vs K | `outputs/20261025-validation-quality-k-sweep-summary-v2/method_summary.csv`, `exact_speedup.csv`, `manifest.json`; source studies `outputs/20261025-validation-quality-k{4,8,16,32}-frozen-policy/`. `N=262,144`, `D=4`, K=4/8/16/32, seeds 0–4, five repeats, two warm-ups, `max_steps=100`, frozen 5% target. | Exact C0/G0 attained 25/25 targets at each K. Non-exact methods attained 15/25 at K=4 and 0/25 at K=8/16/32. Paired exact CPU/CUDA median ratios (IQR) were 0.744× (0.341), 0.930× (0.204), 1.363× (0.211), and 3.283× (1.536). These are single-session, per-K observations; `quality_target` is not Lloyd convergence. |
| Development N/K quality target matrix | `outputs/20261026-real-quality-matrix-completion/individual_runs.csv`, `method_summary.csv`, `manifest.json`; linked source archives are listed in that manifest. D=4, seeds 0–4, max 100 steps, frozen 5% target; N=1,048,548 and 4,194,265, K=4/8/16/32. The summary has methods C0/G0/G1/G2CUDA/G3CUDA/G4CUDA/G5CUDA; other methods have separate source records in selected development archives. | Existing target-hit table: N=1,048,548: K4 exact 10/10 and non-exact 10/10; K8 exact 10/10 and non-exact 2/10; K16 exact 6/10 and non-exact 0/10; K32 exact 4/5 and non-exact 0/5. N=4,194,265: K4 exact 5/5 and non-exact 3/5; K8 exact 5/5 and non-exact 0/5; K16 exact 3/5 and non-exact 0/5; K32 exact 3/5 and non-exact 0/5. Most approximate runs reached `max_steps`; exact also missed some cells. Report paired exact ratios only in the four cells documented as target-reaching comparisons: 1.387×, 2.484×, 3.968× and 5.624× for N≈1.05M K4/K8 and N≈4.19M K4/K8. One-session, low repeat count, nested crops. |
| Held-out quality and method comparison | Ten CPU/CUDA archives summarized in `outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/` (`seed_method_summary.csv`, `aggregate_method_summary.csv`, manifests and ten quality trajectories); all-method raster outputs in `outputs/20261025-test-scene-all-method-maps/`. K=16, N=262,144, D=4, seeds 0–4, five fits/method/seed, two warm-ups, max 1,000 steps, frozen 5% target. | Exact C0/G0 achieved 25/25 hits per execution group; all Mini-Batch/ANN/adaptive methods achieved 0/25 and stopped at `max_steps`. CPU and CUDA ran in separate groups, so report their medians separately; do not form a speedup. This is the untouched test-scene result and must not be used for tuning. |
| Exact real-image scaling | `outputs/20261025-real-fixed-work-scaling-summary-v2/{method_summary.csv,exact_cpu_gpu_speedup.png,real_scaling_median_iqr.png,manifest.json}` and the two N-specific source archives. D=4, K=16, seeds 0–4, two warm-ups, five repeats, one Lloyd update. | Existing paired C0/G0 ratios: 1.087× at N=1,048,548 and 2.052× at N=4,194,265. These are fixed-work, single-session measurements on nested crops from one product, not convergence speedups. Mini-Batch/ANN times are not hardware speedups. |
| GPU ANN feasibility | `outputs/20261025-projection-feasibility-final/{raw.csv,summary.csv,manifest.json}`. Synthetic N=8,192, D=4/14, K=4/8/16/32/64/128/256, seed 0, effort 1/4/16, two warm-ups and five repeats. | Genuine approximate projection search had higher median complete cost than exact GPU search in all tested approximate modes; low-effort recall was 0.8–16.6% across tested D/K. This is a negative feasibility finding for this design/range, not real-scene GPU ANN evidence or a universal ANN result. CPU HNSW real-scene runs are present in real method archives, with no demonstrated benefit at small K. |
| Fixed versus Adaptive V2 observation-budget ablation | `outputs/20261026-adaptive-v2-observation-budget-40m/measurements_reconstructed.csv`, `raw.jsonl`, `manifest.json`, `recovery.json`; interpretation in `docs/ADAPTIVE_V2_BUDGET_ABLATION.md`. Validation 31UFU, N=262,144, D=4, K=16, seeds 0/1/2, same per-seed X/C0, 40,960,000 observation cap, 8,192,000-observation quality milestones, frozen 5% target evaluation-only. | Exact G0 crossed target in 1.085–1.287 s for all three seeds. Fixed decayed and V2 had similar final SSE at equal observation caps; each missed seed 1. V2 completed the cap in one run/seed faster than fixed decay, but runs are unreplicated; it did not approach G0 time-to-target. The original CSV writer failed; the reconstructed measurements are derived from preserved raw records and explicitly labeled. Keep this exploratory. |
| Cumulative versus decayed update diagnostic | `outputs/20261026-validation-decayed-count-k16-b8192-seed0/checkpoint_metrics.csv`, `summary.json`, provenance/checksum files; narrative in `docs/REAL_SCENE_RESULTS.md`. Validation, N=262,144, D=4, K=16, seed 0, same initial centroids and pre-generated batches, 5,000 updates, 8,192 batch, decay half-life 10 passes. | Cumulative SSE 28,276.14 at update 5,000 and did not reach target; decayed SSE 24,703.84 and first crossed at checkpoint 2,100. One matched diagnostic per policy, about 9.14 vs 9.34 s; not evidence of a speedup or a generally superior update. |

## 4. Genuine, synthetic and diagnostic evidence boundaries

**Genuine Sentinel-2 experiments:** the real scene studies named above use
prepared data from the three SAFE products. They support real-image
preprocessing, exact clustering, selected Mini-Batch/ANN/adaptive comparisons,
held-out K=16 quality outcomes and genuine
georeferenced maps. The early `outputs/20261024-real-evaluation/` count of 360
measured fits (and the report's 423 measured/192 warm-up count across its
original archive set) predates the subsequent large matrices and must not be
added to later counts without deduplicating source archives.

**Synthetic-only results:** D=14 scaling, HNSW effort/approximation-quality
sweeps at K=64/256, CUDA projection ANN feasibility, generic CUDA timing and
many unit tests. They demonstrate implementation behavior on controlled data,
not satellite-scene outcomes. Do not use them as real-image figures or
performance claims without clearly labeling them synthetic.

**Diagnostics, not frozen-policy comparisons:** the 5,000-update
cumulative/decayed K=16 replay and Adaptive V2's observation-budget ablation
are validation-only studies. The decay factor and V2 policy are experimental;
the frozen quality target was evaluation-only in V2. They do not replace the
original cumulative Mini-Batch baselines, establish broad convergence, or
support claims about held-out performance.

## 5. Claim audit for the paper

Supported when stated with the cited workload/protocol:

- Exact CUDA was faster than paired exact CPU in some larger real workloads;
  the measured ratios vary by N, K and protocol. At smaller workloads CPU can
  be faster. Cite the exact paired source table, not a generic maximum alone.
- Exact CUDA and CPU produced matching results within the tested numerical
  criteria; final exact SSE/labels were checked in the recorded workflows.
- Fixed/adaptive Mini-Batch and ANN variants often missed the frozen 5%
  objective target under the stated caps. State `max_steps` explicitly;
  do not call these converged runs.
- The tested CUDA projection ANN was slower end-to-end than exact GPU search
  on its synthetic N/D/K range. This is a design/range-specific negative result.
- The one matched validation diagnostic supports cumulative-count inertia as
  a plausible cause in that K=16 case; it does not identify a universally
  optimal forgetting factor.
- One Adaptive V2 observation-budget comparison suggests higher throughput
  than fixed decayed Mini-Batch at similar quality for the tested cap, but
  is unreplicated and remains preliminary. Exact CUDA reached the same target
  much sooner in that comparison.

Not supported / must not be claimed:

- GPU ANN acceleration or fewer real-scene distance evaluations across the
  board.
- A general quality or time-to-target advantage for adaptive methods.
- Convergence for any run whose saved termination reason is `max_steps` or
  `observation_budget`, unless its recorded convergence flag independently
  confirms the criterion.
- Semantic land-cover accuracy, class names, or external validation against
  ground truth; no aligned labeled raster is available.
- Peak GPU/host memory or sustained GPU utilization from session snapshots.
- Per-kernel bottleneck attribution; the Nsight capture did not yield a usable
  trace.
- Robust large-N performance across independent sessions or real D=14 data.

## 6. Publication-ready figure and table selection

### Figure verification corrections (26 October 2026)

The report-assets package at `outputs/report-assets-20261026/` contains eight
byte-identical PNG copies, each opened with Pillow and checked against its
source SHA256. It includes the genuine development composite, held-out true-
and false-color views and G0 cluster-map render, development target-hit chart,
validation paired speedup chart, and one held-out seed-0 CPU and CUDA
trajectory each. The component RGB and cluster images are from the same
held-out 512x512 crop; the associated `clusters.tif` was opened and checked as
512x512, EPSG:32630, 10 m, with labels 0–15. The PNG render itself is a display
asset; the GeoTIFF carries the geospatial transform.

Two interpretation details are material. First, the development hit-rate plot
groups CUDA variants whose per-method hit counts match in each plotted cell;
its denominator is the repeated-fit count for a method, not the sum of fits
across five methods. Describe it as common per-method hit rate, not a pooled
count. Second, `outputs/20261025-real-fixed-work-scaling-summary-v2/exact_cpu_gpu_speedup.png`
is excluded because the plot title says “Mean paired” while the source CSV
reports the **median** paired speedup. Use `real_scaling_median_iqr.png` for
one-update times and cite `exact_speedup.csv` for the median ratio; do not
retouch the existing image.

The selected held-out CPU/CUDA trajectory images are each seed 0, measured
repeat 0, and were collected in separate execution groups. They illustrate
the SSE trajectories but do not establish a paired CPU/GPU time ratio.

Recommended core figures (use existing files; do not combine unlike protocols):

1. **Scene and clusters:** existing real-scene composite plus a same-crop
   RGB/NIR/cluster panel from the exact held-out map paths. State that labels
   are unsupervised and the held-out map is for visualization, not tuning.
2. **Validation target attainment by K:**
   `outputs/20261026-real-quality-matrix-completion/quality_target_hit_rate.png`
   for development N/K behavior, and/or
   `outputs/20261025-validation-quality-k-sweep-summary-v2/`'s existing
   `exact_speedup_by_k.png`. Keep the target-hit and speedup figures distinct.
3. **Held-out K=16 quality-versus-time:** the ten existing
   `quality_seed{0..4}_{cpu,cuda}.png` trajectories under
   `outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/`. Explain
   exact methods reach the target while approximate methods hit the cap.
4. **Exact fixed-work result:** use Table 5 and the exact-only
   `exact_speedup.csv` records. The existing all-method median/IQR chart is not
   selected because it includes methods outside this report scope; an exact-only
   plot can be prepared later from the saved summary rows if needed.
5. **ANN feasibility:** derive a plot later from the saved synthetic
   `raw.csv/summary.csv` only if useful; label synthetic, D/K and index/query
   costs. Do not imply satellite-data GPU ANN.
6. **Adaptive V2/decay:** make a plot later from the saved checkpoint/history
   records only if the paper has space; show equal-observation milestones,
   target as evaluation-only, actual batch schedule and runtime. Label the
   one-session/one-seed-per-method limits.

Recommended result tables:

- Scene/product/crop/bands/valid-pixel counts and split provenance.
- Method/device matrix, including whether each ANN path is CPU HNSW or CUDA
  projection search.
- Validation and development target-hit counts with N/K/seeds/repeats/cap.
- Paired exact C0/G0 speedups with median/IQR and exact workload protocol.
- Negative feasibility/adaptive outcomes and all target misses; separate
  observed results from hypotheses and implementation-only tests.

Every caption should identify scene/split, valid N, D, K, seeds, fit/update
budget, repeats, warm-ups, target definition, timing boundary, hardware and
whether the plot uses measured, summarized or reconstructed rows. Preserve
raw result provenance and do not pool sessions.

## 7. Seventeen-section report outline

Use the exact content in both report variants; change only formatting,
pagination and template-required presentation.

1. **Title** — approved title, team roster as confirmed by faculty.
2. **Abstract** — objective, methods, scenes, strongest measured findings and
   main limitations; no unverified speedup promise.
3. **Keywords** — parallel computing, CUDA, K-Means, Sentinel-2, Mini-Batch,
   approximate nearest-neighbor assignment, adaptive control.
4. **Introduction** — satellite pixel workload and computational motivation.
5. **Literature Review** — systematic 8–10-paper comparison and bounded gap.
6. **Problem Statement** — exact assignment/updates and quality-cost tradeoff.
7. **Objectives** — map to implemented methods and evaluated research
   questions; mark objectives with negative outcomes honestly.
8. **Proposed Methodology** — preprocessing, splits, models, seeds, target,
   controller observables and evaluation design.
9. **System Architecture** — preprocessing/data flow, C0/G0, Mini-Batch,
   CPU/GPU ANN, adaptive controllers, result/map paths.
10. **Algorithms** — Lloyd, cumulative Mini-Batch, ANN assignment/index
    lifecycle, batch/effort/joint control, and separately labeled experimental
    decayed/V2 methods.
11. **Experimental Setup** — WSL/CPU/GPU/software, SAFE products/crops,
    valid N/D/K, seeds/repeats, timing scopes, memory and quality target.
12. **Results** — scene maps, exact CPU/CUDA, target-hit tables, ANN feasibility, adaptive and decayed diagnostics.
13. **Performance Analysis** — paired exact speedups, throughput/overheads and variability; no false method-to-method speedup.
14. **Comparison with Existing Methods** — literature-grounded and
    protocol-aware comparison; do not compare published numbers as a shared
    leaderboard.
15. **Conclusion** — measured conclusions, including negative findings.
16. **Future Scope** — profiler, broader replication, potential ANN and
    adaptation work, labeled reference data if available.
17. **References** — verified bibliographic metadata and cited sources.

Also include the guideline-requested execution screenshots, graphs, tables,
speedup/efficiency analysis where applicable, and an explicit contribution
statement in the presentation/demo package.

## 8. Citation and literature-survey work remaining

`PROJECT_DESCRIPTION.md` lists six references. The manuscript’s verified list
contains eight unambiguous peer-reviewed references from 2022–2025, meeting
the lower end of an 8–10 recent-paper requirement. He et al. (2022 issue,
2021 online-first) is a borderline ninth peer-reviewed source; the 2025 seeded
ANN source is an arXiv preprint and is not counted as peer-reviewed. Zhang et
al. (2021) is retained as supplementary context, but is outside a strict
rolling five-year cutoff as of 26 October 2026. Metadata were checked against
publisher or stable primary records. Before submission:

1. Confirm whether faculty counts publication year or first-online date for
   the five-year cutoff, and whether the preprint may appear in the survey.
2. The current references cover GPU K-Means, Mini-Batch convergence, large-data
   clustering, and remote-sensing clustering. The 2025 seeded ANN work remains
   a preprint; it is relevant to large-K assignment but must not be described
   as peer reviewed. The paper should state that the team’s HNSW/CUDA
   projection experiments are its own feasibility results, not replications
   of that preprint.
3. For each source, record dataset, hardware, parallel model, algorithm,
   reported metric, limitation and relevance. Keep published results separate
   from this project's measurements.
4. Verify every numeric literature claim from the original paper and cite the
   page/table where it appears. Use IEEE-formatted references for the survey;
   apply the target report template's reference formatting consistently.
5. Restate the research gap narrowly after reviewing the additional work; the
   current gap is explicitly provisional.

The manuscript bibliography uses eight unambiguous recent peer-reviewed
papers: Pant et al. (2025), Han and Lee (2024), Li et al. (2023), Zhu et al.
(2024), Guo et al. (2022), Paradis (2022), Schwartzman (ICLR 2023), and
Mussabayev et al. (2023). He et al. (2022 issue; first online 2021) is a
borderline CPU/GPU reference, and Spalding-Jamieson et al. (2025) is an
explicitly labeled preprint. Zhang et al. (2021) is supplementary context.
Full publisher/venue metadata and DOI or stable identifiers are included;
published numeric claims are not imported into project result tables. If any
published numeric comparison is added later, check its source page/table and
hardware/data protocol first.

| Manuscript ref. | Verified publication | Status for recent-paper count | Relevance / retained source |
|---|---|---|---|
| [1] | Pant, Leigh, and Fajardo Rueda, *Algorithms*, 2025, DOI 10.3390/a18080532 | Peer-reviewed recent | Parallel K-Means for satellite-image analysis; retained from project description |
| [2] | Han and Lee, *Remote Sensing*, 2024, DOI 10.3390/rs16010102 | Peer-reviewed recent | Parallel inter-image satellite K-Means; retained |
| [3] | Li, Frank, and Pfahringer, *Data Mining and Knowledge Discovery*, 2023, DOI 10.1007/s10618-022-00869-6 | Peer-reviewed recent | GPU exact/batched K-Means; retained |
| [4] | Zhu et al., *IEEE Transactions on Neural Networks and Learning Systems*, 2024, DOI 10.1109/TNNLS.2023.3279122 | Peer-reviewed recent | Mini-Batch staleness and quality; retained |
| [6] | Guo et al., *Remote Sensing*, 2022, DOI 10.3390/rs14153635 | Peer-reviewed recent | Satellite time-series clustering |
| [7] | Paradis, *International Journal of Applied Earth Observation and Geoinformation*, 2022, DOI 10.1016/j.jag.2022.102675 | Peer-reviewed recent | Large-scale spectral-image unsupervised analysis |
| [8] | Schwartzman, ICLR 2023, OpenReview `jREF4bkfi_S` | Peer-reviewed recent | Mini-Batch termination analysis |
| [9] | Mussabayev et al., *Pattern Recognition*, 2023, DOI 10.1016/j.patcog.2022.109269 | Peer-reviewed recent | Large-data K-Means/decomposition |
| [5] | He, Vialle, and Baboulin, *Concurrency and Computation: Practice and Experience*, vol. 34, no. 14, e6621 (2022 issue), DOI 10.1002/cpe.6621 | Borderline: issue 2022, online-first 2021-09 | CPU/GPU K-Means; supplemental recent reference |
| [10] | Spalding-Jamieson, Robson, and Zheng, arXiv:2502.06163 (2025) | Preprint; not peer-reviewed | Seeded ANN for large K; retained from project description, explicitly labeled |
| [11] | Zhang et al., *Remote Sensing*, 2021, DOI 10.3390/rs13193993 | Supplementary; outside strict current rolling window | Satellite time-series clustering; retained from project description |

The eight peer-reviewed sources marked “recent” satisfy the minimum of eight
without relying on the borderline 2021 online-first paper or the preprint.
The six original project-description references are all retained as [1]–[5],
[10], and [11].

The guideline provides Overleaf links to Springer Nature and IEEE conference
templates. It does not specify Word as the required file format. If final
`.docx` reports are mandatory, confirm that with faculty or obtain approved
Word templates before report generation.

## 9. Original faculty-comment alignment review

The manuscript addresses the explicit proposal comments with the following
evidence and caveats:

- **Title and ANN meaning:** title is retained and ANN is expanded as
  approximate nearest-neighbor assignment. Author roster remains unconfirmed.
- **Sentinel-2 inputs:** product IDs/dates, four bands, mask behavior,
  calibration, standardized split and map provenance are described. Real
  crops used here do not demonstrate broad cloud-mask performance because the
  development reference crop had no excluded cloud/shadow/no-data pixels.
- **Adaptive rules:** the measured policy thresholds and the observed
  validation/held-out outcomes are stated. The policies are deterministic
  heuristics, not evidence that the chosen thresholds are optimal.
- **Moving-centroid ANN index:** CPU HNSW refresh and CUDA projection candidate
  maintenance are distinguished; maintenance and query costs are discussed.
  GPU projection ANN is not represented as GPU HNSW.
- **Fair baseline and ablations:** exact/fixed/adaptive families and
  observation-budget diagnostics use distinct protocols and report unequal
  work explicitly. Capped and target-miss outcomes are retained.
- **Quality/seeds/held-out:** SSE, sampled Silhouette/DBI, repeated seeds and
  a separate test scene are included. No semantic labels or land-cover
  accuracy are claimed.
- **CUDA language:** CUDA kernels are Python-authored Numba CUDA. Instructor
  acceptance of this implementation under the proposal’s C/C++ language
  wording remains open. Original-proposal compliance note (outside the report scope): the proposal
  includes an OpenMP objective, but the final reporting scope explicitly
  excludes it. No fulfillment or acceptance claim is made here. Historical
  source and records remain untouched and are not selected as paper evidence
  or assets.

## 10. Genuine remaining gaps

- Faculty decisions: whether Python-authored Numba CUDA satisfies the original
  C/C++ requirement; confirm official team roster and whether Word reports are
  acceptable in place of the linked templates.
- Literature: the minimum eight recent peer-reviewed sources are present;
  faculty confirmation is needed only for the borderline first-online date,
  preprint treatment, and any stricter literature-analysis expectations.
- Submission deliverables: final reports in both formats, presentation/demo
  materials and team contribution evidence are not present in this repository.
- Publication process: no conference submission receipt or acceptance proof
  has been recorded.
- Evidence limitations to disclose: no labeled land-cover truth; no usable
  Nsight kernel timeline; no systematic peak memory or per-fit utilization;
  large-N repeated-session coverage is limited; no real D=14 study; the
  held-out quality sweep is K=16 only; GPU ANN/adaptive benefits are not
  established.

Do not rerun experiments merely to make these limitations disappear. If the
instructor specifically requires additional evidence, scope and approve that
work separately. Keep `data/raw/` and `outputs/` backed up outside Git; they
are essential to reproducing the paper figures and tables.
