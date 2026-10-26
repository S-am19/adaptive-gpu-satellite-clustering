# PCAP Implementation Plan — Version 2.0 (Two-Developer Workflow)

> **Historical proposal and acceptance baseline.** This document preserves the
> requirements and planned workflow agreed on 20 October 2026; its original
> pre-implementation status statements and future-tense stage descriptions are
> not current project status. See [`docs/IMPLEMENTATION_PLAN_STATUS.md`](docs/IMPLEMENTATION_PLAN_STATUS.md)
> for the audited implementation/evidence state and
> [`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md) for the latest
> real-scene results. Do not delete or silently rewrite this baseline; resolve
> scope changes with the instructor.

**Working title:** Adaptive GPU-Accelerated Mini-Batch K-Means for Satellite Image Clustering  
**Status:** Agreed direction / pre-implementation; no benchmarks or completed code are claimed.  
**Date:** 20 October 2026  
**Primary course:** Parallel Computing  
**Active developers:** Samridhi and Daksh (approximately equal implementation responsibility)  
**Nihal:** **No work assigned at this stage.**  
**Scope adjustment:** Python + custom CUDA kernels first. OpenMP is **deferred, subject to instructor approval**; it remains in the submitted proposal.

---

## 1. Purpose and research question

Build and experimentally evaluate a GPU-based K-Means clustering system that groups *pixels* from real Sentinel-2 Level-2A multispectral images. The system begins with exact sequential CPU and GPU K-Means, then adds fixed Mini-Batch updates, approximate nearest-centroid assignment (ANN = **Approximate Nearest Neighbor**, not an artificial neural network), and a convergence-aware controller which varies batch size and ANN search effort.

**Main question:** When, if ever, does jointly adapting Mini-Batch size and ANN search effort reduce **end-to-end** computation compared with well-optimized exact/fixed CUDA baselines **at a comparable clustering quality**?

**Contribution to demonstrate:** Measured speed/quality/memory trade-offs, algorithmic ablations, and clear explanations of GPU parallel work, overhead and scalability. **A speedup is a hypothesis, not a promised outcome.** In particular, ANN may lose to exhaustive GPU nearest-centroid search when the cluster count K is small.

### In scope
- Real satellite pixel feature preparation, masked multispectral clustering and output maps.
- Exact sequential CPU K-Means; **custom GPU CUDA-kernel** K-Means (assignment + centroid aggregation/reduction).
- Fixed GPU Mini-Batch K-Means with mathematically correct incremental updates.
- ANN centroid assignment **only with a genuine tunable ANN method/index**, updates for moving centroids, and measured query/build/rebuild overhead.
- Separate batch-only, ANN-only and combined convergence adaptation.
- Reproducible tests, comparisons, ablations and honest negative results.

### Deferred / not promised
- OpenMP CPU-parallel baseline (original requirement; obtain approval to defer or implement later).
- A sophisticated web application, deep neural networks, supervised land-cover classification, or blanket claims of improvement on every dataset.
- Treating unlabeled clusters as confirmed land-cover categories without an aligned labeled reference map.

---

## 2. Hardware, operating systems, and tools

| Area | Current choice | Verification needed |
|---|---|---|
| Primary system | Samridhi: Windows 11 + Ubuntu 24.04 WSL2, Intel i5-13450HX, 24 GB RAM, RTX 4050 Laptop GPU / ~6 GB VRAM, CUDA compiler 12.0 detected | **At plan drafting:** custom Python GPU execution and toolkit/bindings compatibility were unverified. They were subsequently verified on Samridhi's WSL system; see `docs/IMPLEMENTATION_PLAN_STATUS.md`. Daksh's separate environment remains unverified here. |
| Second system | Daksh reports similar hardware; ~96.9 GB free on C: and 193 GB on D: | Check exact GPU, RAM, WSL version, Linux CUDA access and Python CUDA installation on his own system |
| Language | Python 3 in a dedicated virtual environment | Choose supported Python/package versions after read-only environment audit |
| CPU reference | NumPy and single-worker Numba `@njit` (or equivalent optimized sequential implementation) | No parallel `prange` in the baseline; warm-JIT timing |
| GPU implementation | Python-defined custom CUDA kernels, **prefer Numba-CUDA if tested compatible** | Verify launch/synchronization, no silent CPU fallback; consider an alternative CUDA-Python solution only if necessary |
| Satellite input | Copernicus Sentinel-2 Level-2A | Select specific product IDs/scenes and download licenses/paths |
| Preprocessing | Rasterio/GDAL, NumPy | Band alignment, masks, scaling and metadata |
| ANN | Investigate FAISS or another practical indexed nearest-centroid library, or a defensible custom indexed search | Confirm actual CPU/GPU availability and whether it improves total time for relevant K; don't assume FAISS HNSW runs on GPU |
| Testing and analysis | pytest, scikit-learn for reference/metrics, Pandas/Matplotlib | Pin versions once working |
| Collaboration | Git/GitHub, Codex CLI inside WSL2 | Both use one repo with feature branches and reviewable changes |

**File placement:** Keep source and environments in each developer's WSL Linux filesystem (e.g. `~/code/pcap-satellite-kmeans`). Large raw data can be stored outside Git on sufficiently spacious drives; record paths rather than hard-code them. Under WSL, benchmark input loading separately if raw data is accessed via `/mnt/c` or `/mnt/d`.

**Do not install GPU drivers, replace CUDA, wipe environments, or upgrade packages without first auditing versions and obtaining approval.**

---

## 3. Ownership: a workable 50/50 split

| Work package | Samridhi (primary) | Daksh (primary) | Joint work |
|---|---|---|---|
| Foundational K-Means | Exact **CPU** baseline, initialization, correctness oracle | Mini-Batch mathematical reference + update tests | Agree on loss and stopping definitions |
| GPU core | Custom CUDA exact assignment, reduction, centroid updates; memory/performance optimization | GPU Mini-Batch sampling/update integration using same assignment/reduction interfaces | Test GPU/CPU numerical consistency |
| ANN | ANN backend selection, index build/rebuild, search-effort API, accuracy and exact-vs-ANN feasibility benchmark | Ensure controller and Mini-Batch call the ANN backend through the common API | Decide operating K range after evidence |
| Adaptive methods | Supply backend telemetry and effort controls | Convergence monitor, batch adaptation, ANN-effort adaptation, rollback/stop rules | Joint ablation experiments |
| Satellite data | Specify required feature contract, validate loaded arrays | Build documented Sentinel-2 preprocessing/loading pipeline | Select scenes and inspect clustered maps |
| Experiments | Core CUDA timing, memory, kernel-level profiler support | Experiment runner, quality metrics, plotting and export | Design fair comparison matrix, independently review results |
| Integration/documentation | Review GPU/ANN changes, explain parallel design | Review Mini-Batch/adaptive changes and experiment wiring | Both merge PRs, test, prepare presentation and final report |
| Nihal | — | — | **No tasks assigned.** |

**Fairness rule:** This is an approximate workload split, not a guarantee of exactly equal coding hours. Rebalance based on measured difficulty rather than giving one developer all the debugging/integration work.

### Avoiding dependency bottlenecks
- **Day 1:** Agree and commit the interface skeleton and small fixture datasets. Neither developer may change public interfaces alone.
- **Samridhi can start independently:** CPU exact implementation, CUDA vector-add smoke test, distance/assignment kernel.
- **Daksh can start independently:** Sentinel-2 loader, Mini-Batch update arithmetic on tiny NumPy arrays, controller state machine and tests using a *mock assignment backend*.
- Once exact assignment is working, Daksh plugs his updates into that backend. Once Mini-Batch is working, Samridhi plugs in ANN through the same backend interface. Neither should have to rewrite the other's algorithms.

---

## 4. Frozen shared data and function contracts — agree BEFORE independent coding

**Dataset arrays**
- `X`: contiguous `float32` array `[N, D]`, one valid pixel per row, one selected spectral band per column. Never assume each row corresponds to adjacent pixels after cloud masking.
- `valid_mask`: `bool` array `[H, W]`; `pixel_indices` maps rows in `X` back to flattened image positions for reconstructing output images.
- `C0`: shared initial `float32` centroids `[K, D]`, saved per seed and shared across compatible algorithms.
- `labels`: integer array `[N]` with values `0..K-1` after final **exact** assignment for evaluation and map production.
- Metadata: scene ID, date, CRS, transform, band names, dimensions, scaling, mask policy, data hashes and development/tuning/test split.

**Logical Python contracts** (adapt interface types as needed for device buffers without forcing GPU↔CPU copying each iteration):

```python
@dataclass
class FitConfig:
    k: int
    seed: int
    max_steps: int
    tolerance: float
    batch_size: int | None = None
    probe_interval: int = 10
    ann_effort: int | None = None

@dataclass
class AssignmentResult:
    labels: object
    squared_distances: object
    query_time_s: float
    index_update_time_s: float = 0.0

class AssignmentBackend(Protocol):
    def prepare(self, centroids, effort=None): ...
    def assign(self, points, centroids, effort=None) -> AssignmentResult: ...
    def refresh(self, new_centroids): ...

@dataclass
class RunResult:
    centroids: np.ndarray
    labels: np.ndarray  # exact assignment of all valid points at completion
    history: list[dict]
    timings: dict
    metadata: dict
```

The real code may use device arrays internally for speed; the logical interface ensures that CPU/GPU/ANN variants produce comparable outputs. Use a separate `fit(...)` entry point for each algorithm but the same configuration and result model. **No full `N × K` distance matrix stored by default.**

### Common algorithm rules
1. Squared Euclidean distance for nearest-centroid assignment; consistent float32 input, documented accumulation precision.
2. Identical seed/C0 for compatible experiments; record method (e.g. seeded K-Means++ or fixed sampled points).
3. Deterministic, documented policy for empty clusters (e.g. retain previous centroid; use the same policy in all comparable variants).
4. CPU full Lloyd iterations update with all valid data; Mini-Batch uses persistent per-cluster counts and incremental updates, not an incorrectly re-used full-batch mean.
5. Fixed validation probe pixels, held aside from batch sampling where appropriate for independent stopping diagnostics.
6. Log every ANN index refresh and its wall time. Final evaluation labels always use an exact full-data pass; report its cost separately and inside end-to-end cost when required.
7. Unit tests own the contracts; version them before parallel development.

---

## 5. Dataset policy and preparation

**Primary:** Sentinel-2 Level-2A surface reflectance, collected from Copernicus Data Space. **At plan drafting,** specific scenes were not yet selected. Three genuine products were later acquired and processed; see `docs/REAL_SCENE_RESULTS.md`. The scene-split and no-test-tuning requirements below remain in force.

### Initial spectral bands and sizes
- Start with B02 (blue), B03 (green), B04 (red), B08 (NIR): all native 10 m bands, so D = 4 for the initial benchmark.
- Record product ID, acquisition time, geographic area, spatial extent, cropped `[H, W]`, band resolution and number of valid pixels.
- Stage datasets: 512×512, 1024×1024, 2048×2048 crops when available; try 4096×4096 only after memory checks. Use 10 m band alignment and resample 20 m SCL *with nearest-neighbor only*.
- Future optional experiments: additional 20 m spectral bands, aligned onto the chosen target grid with documented resampling; distinguish D = 4 from larger D.

### Preprocessing steps and required outputs
1. Select at least one **development** scene, a separate **tuning/validation** scene and one or more **held-out test** scenes. Avoid choosing test scenes based on their final benchmark performance.
2. Load sensor bands and Level-2A scene-classification layer (`SCL`) / suitable masks. Exclude `NoData`, defective/saturated pixels, cloud shadows and cloud/cirrus classes; document exactly which SCL codes were excluded. Do not automatically treat every dark or snow pixel as cloud without a stated policy.
3. Apply product-correct reflectance scaling/offset from metadata, ensure aligned grids, remove non-finite/missing pixels and log valid/removed pixel counts.
4. Choose **one** feature scaling policy (e.g. training-scene derived per-band standardization, or physically scaled reflectance), estimate any statistics using development/tuning inputs only and apply them unchanged to held-out scenes.
5. Create `X`, mask/index mapping and manifest; reconstruct a cluster map only for valid pixels (no invented labels for masked pixels).
6. Put imagery, raw arrays and large plots outside Git. Commit a small synthetic test fixture and dataset acquisition scripts/metadata.

**Evaluation labels:** Unsupervised clusters are cluster IDs, **not verified land-cover classes**. Optional ESA WorldCover or another aligned labeled map may support a *separate* land-cover agreement analysis, never automatic cluster naming.

---

## 6. Detailed build stages, owners and exit gates

### M0 — Repo, architecture and environment (both; Samridhi leads CUDA check)
**Build:** Git repository; `AGENTS.md`; requirements/environment note; common data/algorithm schemas; synthetic fixture; smoke test of a custom Python GPU kernel on both WSL installations; Git ignore rules; branch protection/PR approach.

**Done when:** 2 WSL environments independently launch a tiny **real GPU** kernel and return verified results; versions/hardware captured; tests and static imports run. If Daksh's CUDA is not available, he can still develop CPU mock/controller/preprocessing while installation is resolved separately.

### M1 — Exact sequential CPU and CPU oracle (Samridhi)
**Build:** seeded initialization, assignment, centroid update, stopping, inertia (SSE), empty cluster logic; compiled one-worker CPU baseline.

**Done when:** tiny known datasets pass exact expected assignments; inertia is non-increasing during ordinary full Lloyd updates up to floating-point tolerance; repeated fixed seeds reproduce the same output; timings exclude first JIT compilation or clearly label it.

### M2 — Exact GPU CUDA engine (Samridhi)
**Build:** per-pixel parallel assignment kernel, safe per-cluster sums/counts via atomic accumulation or staged reduction, centroid update, final exact assignment, memory-safe tiled batches for 6 GB VRAM, GPU-resident buffers where possible.

**Done when:** GPU labels/inertia agree with the CPU reference within defined tolerances; no race-related invalid values/empty clusters; 512² and 1024² synthetic runs work; kernel, transfer and end-to-end timings are separated and synchronized.

### M3 — Satellite preprocessing (Daksh, Samridhi reviews)
**Build:** scene discovery/selection, mask application, band alignment, scaling, read/write/loading, metadata manifests, a preview cluster map path.

**Done when:** both implementations consume the same `X`, missing/cloud pixels are excluded, geometry is reconstructable, and datasets from separate scenes are split into development, tuning and untouched test sets.

### M4 — Fixed Mini-Batch K-Means (Daksh)
**Build:** randomized reproducible mini-batch selection, persistent per-cluster sample counts and correct incremental centroid updates; reusable assignment backend; GPU variant. First test with exact NumPy mock, then plug into GPU assignment.

**Done when:** tiny one-step updates match hand calculations; fixed batch experiments (initial candidate sizes 8192, 32768, 65536; revise after memory profiling) run, track number of pixel evaluations and steps, and show stable quality comparisons. Do not equate equal iteration counts across full and Mini-Batch runs.

### M5 — ANN feasibility and backend (Samridhi, Daksh tests compatibility)
**Build:** compare custom exhaustive GPU centroid assignment with a genuine approximate indexed search. Candidate K sweep = 4, 8, 16, 32, 64, 128, 256 (higher K separately if feasible), plus D sweep if meaningful. Test backend availability in WSL; record whether each query is on CPU or GPU. Expose `effort` (e.g. HNSW search depth if chosen), index build/rebuild or update, and assignment recall against exact GPU labels.

**Done when:** index rebuild time and all transfers are measured, recall and search time documented, centroids can be updated safely, and a **go/no-go choice** is recorded: use a beneficial operating K/back-end range, or transparently report the negative finding and discuss expected scope with instructor. Do not assume GPU FAISS supports every CPU index type.

### M6 — Adaptive controller and ablations (Daksh, Samridhi reviews ANN instrumentation)
**Build:** a convergence monitor based on centroid displacement and changed exact assignments on a fixed probe subset; test independently (i) batch-only, (ii) ANN-effort-only, (iii) both. Keep safe minimum/maximum settings, log transitions, provide a quality-based effort rollback, and define stopping independently from approximate assignment stability.

**Proposed initial experimental policy (not a proven optimum):** sample a fixed probe every **10 Mini-Batch steps**; if normalized centroid movement is below `0.001` **and** probe assignment-change rate below `0.005` on **three consecutive probes**, halve batch size down to a selected minimum; for ANN variants reduce backend-specific search effort one supported level (e.g., HNSW `efSearch` 64 → 32 → 16 if that index is chosen). If fixed-probe SSE worsens by more than **1%** relative to its previously retained checkpoint, restore the last safe effort level; evaluate alternative thresholds on validation scenes only. The final policy may differ based on experiments and must be recorded before held-out testing.

**Done when:** decisions can be replayed from logs; adapting batch size alone and ANN effort alone are separately runnable; final quality is evaluated with exact assignments; no premature stopping caused solely by approximate searches.

### M7 — Fair evaluation, profiling and report (both; Daksh leads runner, Samridhi leads kernel profiling)
**Build:** experiment configuration matrix, repeated random seeds (start with 5), runtime/quality/throughput plots, scene and parameter manifests, written limitations, clustered output maps and reproducible analysis scripts.

**Done when:** all ablations below run on development/validation scenes; held-out scenes are untouched until configuration freeze; every reported result links to actual raw logs/hardware/config versions; no fabricated performance or semantic class claims.

### M8 — Original-proposal requirement review and final submission (both)
**Decision gate:** The proposal calls for **sequential, OpenMP and CUDA** K-Means. If the instructor confirms OpenMP can be excluded/deferred, record that approval and the changed scope. Otherwise implement a C++ OpenMP CPU-parallel baseline using the same data/initialization and include fair per-thread-count comparisons (e.g. 1, 2, 4, 8 threads); a Numba `prange` implementation does **not** automatically satisfy an explicit OpenMP requirement. Also confirm that a Python-defined CUDA kernel is acceptable in place of explicitly requested CUDA C++.

**Done when:** submitted work is consistent with the latest instructor-approved scope and limitations are stated.

---

## 7. Required experiment matrix (not optional)

| ID | Method | Updates | Assignment | What it isolates |
|---|---|---|---|---|
| C0 | Compiled sequential CPU K-Means | Full-data | Exact | CPU reference |
| G0 | Custom CUDA GPU K-Means | Full-data | Exact | GPU acceleration vs comparable CPU computation |
| G1 | CUDA fixed Mini-Batch | Fixed batch | Exact | Mini-Batch trade-off |
| G2 | CUDA fixed Mini-Batch + fixed ANN | Fixed batch | Fixed ANN effort | Adding ANN, including index overhead |
| G3 | CUDA adaptive Mini-Batch | Adaptive batch | Exact | Batch adaptation alone vs G1 |
| G4 | CUDA fixed Mini-Batch + adaptive ANN | Fixed batch | Adaptive ANN effort | ANN-effort adaptation alone vs G2 |
| G5 | CUDA combined adaptive | Adaptive batch | Adaptive ANN effort | Interaction/combined effect vs G2/G3/G4 |
| O0 (if approved/required) | C++ OpenMP K-Means | Full-data | Exact | CPU-thread scalability vs C0 |

**Fairness design:** identical inputs, same C0 arrays/initialization for compatible runs, seed control, equivalent evaluation checkpoints and end conditions. Distinguish *same work* comparisons (C0 vs G0) from *time-to-specified-quality* comparisons (e.g. G1 vs G5). Do not present GPU speedups over CPU from an algorithm doing less work as purely hardware parallel speedups.

**Workload matrix:** pixel sizes 512², 1024², 2048², optionally 4096² after memory checks; K at minimum 4, 8, 16, 32 plus higher K for ANN crossover tests; D = 4 initially and larger D if more bands are added. Random seeds = initially 5 across tuning and test, with results reported as mean and spread, not only best run.

---

## 8. Measurements and correctness criteria

### Parallel computing / performance
- **Wall-clock training time** and end-to-end time: decoding/preprocessing separately; include transfers, ANN index build/update, sampling, monitoring, final label assignment as appropriate to the clearly named time scope.
- **GPU kernel time**, synchronization overhead, transfer time, ANN query + index-refresh time, total update steps, data points processed and estimated/recorded distance evaluations.
- CPU/GPU speedup for the **same exact computation**; GPU throughput (pixels/sec), peak host/GPU memory, practical batch memory constraints, utilization/profile diagnostics.
- If OpenMP is implemented: scaling by threads, speedup and CPU parallel efficiency. Do not apply CPU-style `speedup / GPU thread count` as a meaningful GPU efficiency claim.
- Explicitly distinguish compilation/warm-up overhead from steady-state and optionally report one-time start-up overhead. Use `cuda.synchronize()` or accurate CUDA events for device work before stopping timings.
- Report results on **each machine separately**, especially if configurations differ; do not combine cross-machine timings in one speedup ratio.

### Clustering quality
- Full-data or reproducibly sampled **K-Means inertia/SSE** (sum of squared distances to assigned final centroids); normalized error `SSE / N` for comparisons. Optional RMSE defined explicitly as `sqrt(SSE / (N*D))`.
- Silhouette Score and Davies–Bouldin Index on the **same fixed sampled points** to avoid quadratic memory blowups.
- ANN assignment recall/agreement versus exact nearest-centroid labels, with tie-handling policy stated.
- Quality-versus-time graphs and **time to a documented common quality threshold**, plus convergence stability/premature-stop checks; mark algorithms that never meet the threshold.
- Optional external land-cover agreement only if a correctly aligned label source exists. No unvalidated land-cover accuracy claims.

### Evaluation split and reporting
- Tune K, batch ranges, ANN effort, adaptation triggers and allowed quality degradation on **development/validation scenes**.
- Freeze hyperparameters and code before held-out final-scene evaluation. Preserve scenes and seeds in a manifest.
- Save machine-readable per-run JSON/CSV (timestamp, Git commit, data hash, hardware, driver, toolkit, packages, algorithm settings, times, objective, validity, warnings).

---

## 9. Repo layout and Git workflow

```text
pcap-satellite-kmeans/
├── AGENTS.md                     # Codex constraints
├── PROJECT_DESCRIPTION.md
├── IMPLEMENTATION_PLAN.md        # This file
├── pyproject.toml / requirements.txt
├── README.md
├── configs/                      # declarative experiment configs
├── data/
│   ├── README.md                 # acquisition and provenance
│   ├── raw/                      # gitignored
│   └── processed/                # large files gitignored
├── src/pcap/
│   ├── contracts.py              # SHARED: agreed public APIs
│   ├── initialization.py         # Samridhi
│   ├── kmeans_cpu.py             # Samridhi
│   ├── kmeans_cuda.py            # Samridhi
│   ├── assignments_exact.py      # Samridhi
│   ├── assignments_ann.py        # Samridhi
│   ├── minibatch.py              # Daksh
│   ├── adaptive_controller.py    # Daksh
│   ├── preprocessing.py          # Daksh
│   └── metrics.py                # Daksh; reviewed by Samridhi
├── experiments/
│   ├── run_suite.py              # Daksh, reviewed together
│   ├── benchmark_cuda.py         # Samridhi
│   └── plot_results.py           # Daksh
├── tests/
│   ├── fixtures/
│   ├── test_contracts.py
│   ├── test_exact.py
│   ├── test_minibatch.py
│   ├── test_ann.py
│   └── test_adaptive.py
├── outputs/                       # gitignored run results/figures
└── docs/                          # decisions, references, test evidence
```

**Branches:** `main` always passes the minimum tests; `feature/cuda-exact` (Samridhi), `feature/ann` (Samridhi), `feature/dataset` (Daksh), `feature/minibatch` (Daksh), `feature/adaptive` (Daksh); split further as needed. Make small PRs, request the other person's review and avoid simultaneous edits to shared contracts. Neither person commits downloaded imagery, secrets, virtualenvs, large results, or local absolute file paths.

### Codex task protocol (both developers)
1. Read `AGENTS.md`, project plan and current branch; state exact files intended to change.
2. Implement **only one milestone-sized change**; explain the algorithm and CUDA concepts that matter rather than silently generating the entire application.
3. Run relevant tests on the developer's actual machine, report the command, observed results, device used and any failures; never claim GPU execution without verifying it.
4. Avoid package installation/upgrade, dataset downloads or destructive filesystem operations unless a developer approves.
5. Propose commit message; merge only after contract and tests check out. Final integration is a joint responsibility.

---

## 10. Sequence / parallel work calendar (illustrative, not a fixed deadline)

| Period | Samridhi | Daksh | Shared gate |
|---|---|---|---|
| Sprint 1 | GPU smoke test; exact CPU foundation | Repo interface + synthetic Mini-Batch reference; metadata/data scouting | Freeze data/results/backend contracts |
| Sprint 2 | Custom CUDA assignment/reduction | Sentinel-2 preprocessing pipeline; Mini-Batch numerical tests | Same synthetic fixtures & real `X` load |
| Sprint 3 | Exact GPU correctness/performance | GPU Mini-Batch integration | C0/G0/G1 run on one reference scene |
| Sprint 4 | ANN library feasibility, moving-centroid rebuild tests | Controller with mock ANN and exact GPU probes | ANN go/no-go / measured overhead |
| Sprint 5 | ANN integration and accuracy profiler | Adaptive batch and ANN effort variants | G2/G3/G4/G5 integrate |
| Sprint 6 | CUDA performance optimization and memory profiling | Repeated experiment runner, metrics, plots | Frozen configs + full test matrix |
| Sprint 7+ | Review/report/optional OpenMP | Review/report/optional OpenMP | Teacher scope check; held-out results; presentation |

Sprints denote **work packages**, not guaranteed one-week commitments. Shift them based on measured task complexity and available time.

---

## 11. Instructor-comment compliance checklist

| Teacher feedback | Planned evidence | Owners |
|---|---|---|
| Simplify title; ANN definition | Title and expanded term in report/README, no artificial neural networks | Both |
| Specify Sentinel-2 data and preprocessing | Named product IDs/scenes, bands, pixel dimensions, SCL mask codes, band alignment, scaling manifest | Daksh (primary), Samridhi validates |
| Specify precise adaptation triggers and step sizes; detect harm | Versioned thresholds, logged control transitions, rollback, quality/time-to-target comparisons | Daksh (primary) |
| Moving centroids and ANN indexing; small K comparison | `refresh/rebuild` implementation, index update timing, recall, K sweep vs exact GPU | Samridhi (primary) |
| Fair baselines / independently isolate gains | C0/G0/G1/G2/G3/G4/G5 matrix, shared C0/seeds, time-to-quality comparisons | Both |
| Quality errors, repeated seeds, held-out scenes, references and caution about land-cover labels | SSE/RMSE, Silhouette/DBI, five seeds (initial), validation/test split, complete bibliography and careful cluster labels | Both |

**Additional original-proposal requirement:** Sequential, **OpenMP** and CUDA are named explicitly. Deferral is not equivalent to completion; log the instructor's approval or later implement O0. The proposal also mentions C/C++; verify whether Python-authored CUDA kernels are acceptable as the principal parallel programming implementation.

---

## 12. Risks, decision gates and fallbacks

| Risk | Early test | Response |
|---|---|---|
| Python CUDA package incompatible with environment | GPU vector-add kernel on each WSL machine | Pin compatible versions in a venv; evaluate an alternative Python CUDA backend only with agreement |
| CUDA K-Means incorrect or slow | Compare numerical results and profiler on synthetic data | Fix race conditions, aggregation, memory layout before adding features |
| 6 GB GPU memory exhausted | Peak allocation and batch-size tests | Tile inputs and keep only necessary device arrays; shrink working crops |
| ANN loses to exact GPU for satellite K | K/feature dimension sweep including rebuild and transfer cost | Document crossover or negative result; do not fake improvements; discuss scope implications |
| Adaptive policy stops too early | Independent fixed-probe objective and full exact final evaluation | Roll back effort; change policy on validation scenes; prevent early stop on ANN-derived signal alone |
| Developers block one another | Mock assignment, frozen contracts, synthetic fixtures | Parallelize CPU/mock work; review interface changes explicitly |
| Inconsistent performance comparisons | Primary benchmark machine and saved manifests | Run matched workloads and keep hardware-specific results separate |
| Instructor insists on OpenMP/CUDA C++ | Ask before final scope is frozen | Implement C++ OpenMP baseline / agreed language component if required |

---

## 13. Final deliverables and acceptance definition

**Code:** Reproducible preprocessing; exact sequential and CUDA K-Means; fixed Mini-Batch; ANN search or a documented feasibility finding; adaptive batch and ANN controllers; shared tests and parameterized experiment runner; optional approved OpenMP extension.

**Results:** Genuine cluster maps (unlabeled IDs); CPU/GPU comparisons; K-and-N scaling; quality metrics; all independent ablations; logs including ANN update time and end-to-end overhead; mean and spread over repeated seeds; negative outcomes included.

**Documentation:** README with one-machine setup, design/ownership, algorithm explanations, fair benchmarking methodology, limitations, bibliography and a summary of teacher feedback; Git commit references and reproducibility details; presentation/report aligned with actual run outputs.

**Definition of done:** Every claimed implemented feature has passing tests, every reported performance figure can be traced to a real run and configuration, and the final submission matches the instructor-approved scope. **Do not treat the project as completed merely because a demo generates a colored satellite image.**

---

## 14. Immediate next actions — no coding beyond smoke tests yet

1. Both developers review and approve **Sections 3–4**, particularly the assignment backend and `RunResult` contracts.
2. Verify Daksh's precise Ubuntu/WSL Python CUDA execution, rather than assuming identical hardware equals identical software.
3. On each machine perform **read-only checks** for Codex, Python, Numba, CUDA and Git; do not install anything until conflicts and missing pieces are identified.
4. Create a shared GitHub repo in the Linux filesystem, add `AGENTS.md`, this plan, small fixtures and `.gitignore`, and agree branch naming.
5. Test a real custom CUDA vector-add kernel. **Only then** begin M1–M3 in parallel.

### Reference entry points
- Teacher-provided annotated project proposal (the six comments are implemented in Section 11).
- Pant et al. (2025): https://doi.org/10.3390/a18080532
- Han & Lee (2024): https://doi.org/10.3390/rs16010102
- Li, Frank & Pfahringer (2023): https://doi.org/10.1007/s10618-022-00869-6
- Spalding-Jamieson et al. (2025): https://arxiv.org/abs/2502.06163
- FAISS K-Means documentation: https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering
- Numba-CUDA: https://nvidia.github.io/numba-cuda/
- Copernicus Browser: https://browser.dataspace.copernicus.eu/
