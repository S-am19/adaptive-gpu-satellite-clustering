# adaptive-gpu-satellite-clustering
GPU-accelerated adaptive Mini-Batch K-Means for satellite image clustering using Python and CUDA

The original [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) is retained as
the proposal and acceptance baseline; its pre-implementation status wording is
historical. Current gates and evidence are summarized in
[`docs/IMPLEMENTATION_PLAN_STATUS.md`](docs/IMPLEMENTATION_PLAN_STATUS.md), with
chronological real-scene results in [`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md).

## CUDA smoke test (M0)

With the project's Python environment installed in `.venv`, run from the
repository root:

```bash
bash scripts/run_cuda_tests.sh
```

The launcher detects Ubuntu WSL2 and prepends `/usr/lib/wsl/lib` to the child
process's `LD_LIBRARY_PATH`, selecting the Windows-provided CUDA driver bridge.
On native Ubuntu it preserves normal driver-library discovery. It changes no
system configuration or global shell settings and requires no environment activation.

The test disables CUDA simulation, selects the first visible NVIDIA CUDA GPU,
prints its name and compute capability, and verifies 65,539 float32 additions
after launching and synchronizing a custom Numba-CUDA kernel. Missing hardware
or incorrect results fail the test. Any compatible NVIDIA CUDA GPU can be used;
there is no GPU-model restriction.

To select a particular GPU or pass additional pytest arguments:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cuda_tests.sh
bash scripts/run_cuda_tests.sh tests/test_cuda_vector_add.py -v
```

Pytest and Python bytecode caches are disabled by the launcher. This smoke test
verifies GPU execution and correctness; it is not a performance benchmark.

## Shared result contract (M1)

`pcap.contracts` requires NumPy but does not import Numba or CUDA. With the current
source layout, ordinary Python scripts can import it using `PYTHONPATH=src`.
All fit implementations return the same five `RunResult` fields:

- `centroids`: final host `float32[K, D]` NumPy array.
- `labels`: host `int32[N]` labels from exact nearest-centroid assignment of
  **all input points against the final centroids**, including on step-limit exit.
- `metadata`: must include `final_exact_sse`, the finite, nonnegative full-data
  sum of squared distances from that same final exact evaluation. `result.sse`
  reads this value and raises `ValueError` if it is absent or invalid.
- `history`: algorithm-specific progress/probe records. These may be sampled or
  approximate in future variants and never supply `result.sse`.
- `timings`: CPU `compute_s` includes the final exact pass;
  `final_assignment_s` measures that pass and its SSE reduction separately, as a
  **subset** of `compute_s` and `wall_s`. Do not add it to either total. Cold
  timings include JIT compilation; benchmark only with explicit warm-up.

Daksh's future Mini-Batch/adaptive implementations must perform the final exact
evaluation and populate `metadata['final_exact_sse']` even when their last history
entry is a probe. SSE uses squared Euclidean distance; CPU accumulation is float64.
Preserve the returned centroid/label arrays and final metric together as one result.

Run CPU tests without a GPU, or the complete suite with real CUDA access:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_exact.py
bash scripts/run_cuda_tests.sh tests
```

## Exact CUDA K-Means (M2)

`from pcap.kmeans_cuda import fit_cuda` provides `fit_cuda(X, C0, config)` with
the same host inputs, validation, convergence rule and `RunResult` as `fit_cpu`.
CUDA is confined to the optional module; CPU modules and shared contracts remain
usable without a CUDA import. Run the complete real-device suite with:

```bash
bash scripts/run_cuda_tests.sh tests
```

Custom kernels assign one point per thread (with grid-stride tiles), accumulate
float64 cluster sums/int64 counts atomically, and update float32 centroids while
retaining empty clusters. Two-stage block reductions compute SSE and maximum
squared centroid movement. Exact ties retain the lowest centroid index. Atomic
sum order and GPU arithmetic can differ from CPU arithmetic; nearly tied points
can therefore differ within floating-point precision. No speedup is claimed.

The dataset, centroids, labels and working arrays stay on-device. Only scalar SSE
and movement diagnostics are downloaded per iteration; full labels and centroids
are downloaded at completion. No `N × K` distance matrix is allocated. Storage is
approximately `4ND + 12N + 12KD + 16K` bytes plus bounded reduction scratch.
An 80%-of-free-memory preflight guard rejects oversized inputs; out-of-core input
streaming is not implemented. Float64 atomic sums require compute capability 6.0
or later; the GPU model is not hardcoded.

Timing scopes overlap intentionally:

- `wall_s` covers validation, context setup, allocations, transfers, fitting and
  metadata collection up to result construction.
- `compute_s` covers the Lloyd loop including initial/final exact assignments,
  scalar transfers, host monitoring and synchronization.
- `final_assignment_s` covers the last exact assignment, SSE reduction and scalar
  download once; it is already inside `compute_s` and `wall_s`.
- `kernel_s` sums synchronized CUDA-event intervals around kernel groups. Cold
  intervals can include host JIT compilation stalls; these are not isolated
  per-kernel profiler measurements.
- `upload_s` includes initial input/centroid allocation and copying and is within
  `setup_s`; `scalar_transfer_s` and `download_s` measure diagnostic and final
  result downloads. `transfer_s` sums those three transfer scopes and must not be
  added to totals that already include them.

Runs record the actual GPU, compute capability, CUDA driver/NVRTC versions,
Python/package versions, configuration, reduction policy and estimated buffer
bytes. Warm kernels explicitly before any later steady-state benchmarking;
atomic contention, float64 throughput and monitoring overhead remain unoptimized.

## Synthetic CPU/CUDA pilot benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/benchmark_cuda.py
```

The runner configures the WSL2 driver path by re-executing its own process when
needed; native Ubuntu keeps normal driver discovery. It uses real CUDA hardware.
Defaults are N=1,024/8,192, D=4/14, K=4/16, seeds 0/1/2, one complete warm-up per
method per case, then five measured repeats. Each pair uses identical float32
arrays, sampled initial centroids and configuration (`max_steps=30`, `tolerance=1e-6`).
Dataset sizes increase in order, with conservative RAM/VRAM checks and a 120-second
budget checked between repeats/cases. This budget cannot interrupt an active fit.
Larger experiment matrices require a separate decision; defaults are only a pilot.

External `perf_counter` measurements cover the same complete `fit()` boundary,
including validation, allocations, final labeling, metadata and GPU transfers.
Previous GPU work is synchronized before either timer; `fit_cuda` itself completes
device work before returning. Generation, initialization, output verification,
explicit `gc.collect()` and file writes are outside both timers. Automatic garbage
collection remained enabled during the completed pilot and may run inside timed
fits. The runner does not disable it; future manifests record its initial state.
Method order alternates
per repeat, with seed-dependent starting order. Warm-up rows are saved but excluded
from summaries. Small-grid occupancy hints alone are suppressed during the runner;
other warnings are recorded. No algorithms or instrumentation are altered.

Raw CSV, per-case summary CSV, and a JSON manifest are saved in a unique directory
under `/tmp/pcap-benchmarks/` by default. `--output-dir PATH` selects a new directory;
keep generated results outside Git. Artifacts include seeds, data/C0 hashes, source
hashes, Git state, hardware/tool versions, warnings, final SSE, iterations and
termination reasons. Summaries report medians, quartiles, IQR, standard deviation
and min/max across repeats **separately for each seed**.

Future schema-version-2 raw rows also record final centroid displacement and
`convergence_satisfied`, calculated directly as displacement <= tolerance. Each
method's summary records iteration counts, termination reasons, whether all repeats
satisfied the criterion, how many did, and maximum final displacement. Reaching
`max_steps` alone does not prove non-convergence: a run can satisfy tolerance on
that last step. `quality_valid` means the pair passed quality/matched-work checks;
it does not mean convergence. Existing pilot CSV/JSON files are not rewritten.

End-to-end speedup is CPU external median / GPU external median. Internal GPU
`compute_s`, CUDA-event intervals and transfer-related timings are separate
diagnostics with overlapping scopes; they are not added together or used as an
end-to-end speedup denominator. In particular, GPU `compute_s` includes host
monitoring/scalar transfers and `kernel_s` includes inter-launch gaps. Full-fit
timings retain instrumentation and metadata overhead, and exclude process startup
and JIT warm-up. Warming each case also warms data/allocator caches; this is not a
cold-start or pure-kernel benchmark. Clock/thermal variation is not controlled.

Every pair is checked outside timing for final exact labels/SSE, CPU/GPU SSE and
centroid agreement, identical iteration counts/termination and unchanged inputs.
A failed gate is saved and stops the run; no speedup is reported for invalid pairs.
The three-seed pilot does not replace the plan's larger five-seed final experiments.

The original main pilot is preserved byte-for-byte under
`~/pcap-benchmark-archive/20261020T174112.839285Z/`, with a SHA256 inventory and
separate provenance notes. Its pre-warning-fix runner was reconstructed and verified
against the saved source hash. The archive also includes the associated algorithm
sources and a **post-run** dependency snapshot; unrecorded original transitive
package versions cannot be established retrospectively. Keep the original manifest
and CSVs unchanged when copying archives; record later annotations separately.

An audit of the original raw rows confirms two limit-terminated cases:
`n8192_d14_k4_seed0` and `n8192_d14_k4_seed1`. Each has five CPU and five GPU
measurements at 30 steps: 20 measured rows total. There is no discrepancy with the
earlier report of two cases. Historical non-convergence follows the recorded
termination reason; those original rows did not save final centroid displacement.


## Sentinel-2 Level-2A preprocessing

`pcap.preprocessing` prepares raw product DN for B02, B03, B04 and B08 on a
common 10 m grid. No imagery is bundled or downloaded. Both array and file preprocessing use
**Rasterio** for authoritative GDAL/PROJ CRS validation. Rasterio is imported
when preprocessing grids/files are used; existing CPU/CUDA modules remain
independent. Rasterio 1.5.1 is installed in the project environment with bundled
GDAL; no system GDAL changes are made by this module.

The observed, tested project `.venv` contains these exact versions:

```text
rasterio==1.5.1
affine==3.0.1
attrs==26.1.0
certifi==2026.10.19
click==8.5.0
pyparsing==3.3.3
```

Rasterio reports bundled GDAL **3.12.4** in this environment. These versions
record the validated environment for reproducibility; they are not mandatory
application version checks or a requirement to install system GDAL 3.12.4.
Preprocessing requires compatible Rasterio functionality for CRS parsing and
raster I/O; compatibility with other package versions has not been validated.

```python
from pcap.preprocessing import load_sentinel2, fit_standardization

# Supply assets from the same original SAFE product/granule, not already scaled
# or harmonized STAC assets. metadata_path is the product MTD_MSIL2A.xml.
scene = load_sentinel2(band_paths, scl_path, metadata_path,
                      window=(0, 0, 1024, 1024), sample_size=None, seed=0)
# Fit once on TRAINING scenes only; apply the saved statistics to every split.
stats = fit_standardization([scene.X], training_ids=["training-product-id"],
                            split="train")
normalized = scene.standardized(stats)
X = normalized.X  # finite, contiguous float32 [N, 4]
# After clustering: normalized.map_labels(result.labels), masked/unsampled=-1.
# Persist stats.to_json() and normalized.metadata with the dataset artifacts.
```

Reflectance is `(DN + BOA_ADD_OFFSET[band]) / BOA_QUANTIFICATION_VALUE`, parsed
from product XML. Processing baseline >=04.00 requires explicit offsets;
earlier products without offsets use zero. DN=0, metadata special values,
raster masks, nonfinite values and invalid SCL pixels are excluded before
feature construction. Valid negative reflectance is retained. Calibration
metadata and its SHA256 are recorded; no fixed 1/10000 conversion is assumed.

SCL is georeferenced onto the band grid with **nearest-neighbor** resampling.
The array API `preprocess_arrays` requires explicit north-up same-CRS grids and
handles crop offsets; it never guesses alignment from array shape. The file reader reads a bounded SCL source window with its masks applied, fills
invalid categories with no-data, then reprojects with nearest-neighbor resampling.
This preserves explicit dataset masks even when source no-data is also present.
It checks common band grids/CRS.
Default SCL exclusions are 0 (no-data), 1 (defective/saturated), 3 (cloud shadow),
8/9 (cloud), and 10 (cirrus). Snow/ice (11) is retained unless `exclude_snow=True`;
dark (2) and unclassified (7) pixels are retained. SCL is a quality mask, not
verified land-cover labels.

Crop windows are `(row, column, height, width)` and file reads default to at most
1024*1024 crop pixels. Sampling is seeded, without replacement, after masking;
it reduces X but does not replace cropping as the raster-memory bound.
`valid_mask` records all eligible crop pixels, while `pixel_indices` identifies
selected X rows in flattened crop coordinates. Crop/source transforms and CRS
support reconstruction; unsampled pixels remain unassigned.

Standardization uses pixel-weighted float64 training means/population variance;
constant bands use scale 1. Transform never refits statistics. The explicit
training split and scene IDs help prevent accidental leakage, but callers must
supply truthful split assignments. Saved statistics include their training IDs.

Run synthetic tests without satellite data:

```bash
.venv/bin/python -B -m pytest tests/test_preprocessing.py -q -p no:cacheprovider
```

The synthetic preprocessing suite requires Rasterio for CRS validation, including
tests that pass NumPy arrays. The tests do not download satellite data. A genuine
baseline-05.10 SAFE/JP2 product was separately validated; see
`docs/REAL_SCENE_RESULTS.md`.

Masked NumPy band arrays retain their masks, combined with explicit band masks,
DN/special-value checks and SCL exclusions. Masked SCL cells become no-data
before nearest-neighbor alignment; masked entries in explicit validity masks
are also treated as invalid. Input arrays and masks are not modified.
Both array and file APIs parse CRS definitions through Rasterio and require
projected coordinates with recognised metre-based horizontal units and unit
conversion factor 1. Geographic, feet-based and unknown CRS/unit definitions
are rejected; 10 m affine pixel size is checked separately. Equivalent EPSG/WKT
definitions are compared by parsed CRS equality, not raw string matching.

## Fixed CPU Mini-Batch K-Means

`pcap.minibatch.fit_minibatch_cpu(X, C0, config)` returns the existing `RunResult`
using exact sequential CPU batch assignment. It has no CUDA, HNSW or Rasterio
imports and does not modify the full Lloyd CPU/CUDA baselines.

```python
from pcap.contracts import FitConfig
from pcap.minibatch import fit_minibatch_cpu

config = FitConfig(k=len(C0), seed=0, max_steps=100, tolerance=1e-6,
                   batch_size=32, probe_interval=10)
result = fit_minibatch_cpu(X, C0, config)
```

`batch_size` must be in 1..N and `ann_effort` must be None. The RNG draws a
uniform sample without replacement inside each update; rows may recur across
updates. For batch_size=N, all input rows are used in input order. A separate
seeded RNG chooses the fixed stopping probe, so changing probe cadence does not
change batch draws. Seeds reproduce results within the recorded NumPy/Numba
environment; initial centroids remain caller-provided and are not observations.

Float64 sums and int64 per-cluster observation counts persist across updates.
Every point is assigned against the pre-update centroids. For a cluster with
new observations, its centroid becomes the cumulative observation sum divided
by cumulative count, rounded to float32. Clusters receiving no batch points
retain their centroids. This is a running Mini-Batch update, not ordinary Lloyd
K-Means repeated on samples; even batch_size=N differs from Lloyd after the
first update because observation counts persist.

Stopping checks occur every `probe_interval` updates. Three consecutive checks
must have maximum centroid movement since the previous checkpoint <= tolerance
and unchanged exact labels on the fixed probe of up to 1024 input rows. The
probe overlaps training data: this is a heuristic stopping diagnostic, not
held-out validation or proof of full-data convergence. If the cap arrives first,
termination is `max_steps`; an unchanged single random batch cannot stop a fit.
An unstable checkpoint resets the consecutive stable count. If the third stable
checkpoint occurs exactly at `max_steps`, termination is `tolerance`; reaching
the cap without that third check remains `max_steps`. These reasons describe
the probe heuristic, not guaranteed full-data convergence.

History records pre-update `batch_sse_before_update`, batch movement, cumulative
observations and, at checkpoints, post-update `probe_sse`, checkpoint movement
and probe label changes. Neither objective is full-dataset SSE, and neither is
required to decrease. At completion, one exact full-data assignment sets labels
and `metadata['final_exact_sse']`, which is the value exposed by `result.sse`.

Sampling, batch assignment, accumulation/update, monitoring and final assignment
have separate host timings. They are subsets of `compute_s`, which is inside
`wall_s`; do not add them again to compute/wall. Cold assignment/accumulation
calls include Numba JIT. No performance or GPU Mini-Batch claim is made. Metadata
records stopping, sampling, probe indices, counts and point-processing work;
Mini-Batch steps are not equivalent to full Lloyd iterations.

## Pure adaptive batch-size controller

`pcap.adaptive_controller` supplies immutable `BatchPolicy`, `ControllerState`,
`Checkpoint` and `BatchDecision` records, `initial_state(policy, objective)` and
pure `transition(policy, state, checkpoint)`. It does not execute K-Means,
consume random numbers, measure time or decide convergence. The separate CPU
adaptive integration is described below; the fixed baseline remains unchanged.

Use `BatchPolicy.for_run(N, config.batch_size)` to default the minimum to
`min(initial_batch, 256, N)` and maximum to `max(initial_batch, min(N, 65536))`.
Explicit bounds must include the initial batch and lie within 1..N. Feed exact
fixed-probe mean SSE, normalized checkpoint movement and changed-label fraction.
The CPU integration checks every `config.probe_interval` updates (default 10),
normalize movement by the fixed initial probe RMS spread `s` (floored at 1e-12),
and set `objective_floor=1e-12*s*s`. The standalone floor defaults to 1e-12.

Priority is: probe objective more than 1% above its retained best checkpoint
increases batch size to at least double or the associated safe batch; otherwise
three alternating relative objective changes, each strictly greater than 0.001
in magnitude, double it; otherwise three quiet checkpoints halve it. Quiet means
normalized movement <=0.001, label-change fraction <=0.005 and absolute relative
objective change <=0.001. Relative changes divide by the larger of the previous
objective and the positive objective floor. All thresholds and quiet-streak
length are configurable experimental choices. Changes clip to bounds; halving
uses integer floor. Lower probe objectives update the safe objective and the
batch that produced it. No centroid/statistic rollback is requested.

Actual batch changes reset quiet/oscillation history. Clipped requests are
logged with `changed=False` and do not trigger these resets, including repeated
deterioration at maximum batch size. An unquiet checkpoint still clears the
quiet streak normally. Future callers must distinguish a requested action from
an actual size change and must not reset stopping counters on clipped no-ops.
Maximum-batch stopping confirmation **freezes adaptation** in the CPU
integration; it does not invoke transitions or change batch size during that
confirmation phase. Controller decisions are reproducible from policy, initial
state and checkpoint signals. Probe diagnostics do not guarantee full-data
convergence or detect every rare cluster; policy tuning requires later validation.

## CPU adaptive Mini-Batch integration

`pcap.adaptive_minibatch.fit_adaptive_minibatch_cpu(X, C0, config, *, policy=None)`
returns the existing `RunResult`. It uses exact CPU assignments and the committed
pure controller; fixed Mini-Batch and the shared contracts are unchanged. Supply
an optional `BatchPolicy` whose initial size equals `config.batch_size` and whose
maximum is <=N. With no policy, bounds include the initial size and the objective
floor is `1e-12*s*s`, using the initial fixed-probe RMS spread `s` (floor 1e-12).
An explicitly supplied policy retains its configured objective floor.

Float64 sums, int64 counts, float32 centroids and the batch RNG persist through
size changes. Sampling is uniform without replacement within each update; later
updates may revisit rows. Full batches use input order. A separate seeded stream
chooses up to 1024 fixed probe rows. Every `probe_interval` updates, exact probe
mean squared distance, normalized checkpoint displacement and label-change
fraction feed the controller. Decisions take effect on the next update, with
safe-reference and before/after controller state logged for replay. Clipped
requests are not actual changes. Observation work is the sum of actual batch
sizes, including confirmation updates, not steps times the initial batch size.

Three successive checkpoints with absolute displacement <=`config.tolerance`,
unchanged exact probe labels, no deterioration and no real controller size
change request confirmation. The next update uses the maximum batch size.
Adaptation is then frozen: the transition function is not called. Three fresh
qualifying checkpoints at that size are required to stop. Confirmation also
requires probe objective no more than the policy deterioration allowance above
its entry value. An unstable/quality-failing checkpoint resets the confirmation
counter; the run remains frozen at maximum size until it qualifies or reaches
the cap. Even when already at maximum, fresh confirmation checks are required.
A third qualifying confirmation exactly at `max_steps` reports `tolerance`;
otherwise an exhausted cap reports `max_steps`, never unconfirmed convergence.

History separates batch/probe objectives, controller requests, actual size
changes, confirmation phase/counters and observation work. Final full-data labels
and `result.sse` always use exact assignment at returned centroids. Host timings
separate sampling, batch assignment, update, monitoring, controller/confirmation
bookkeeping and final assignment; all are subsets of compute and wall time.
Cold timings include JIT. No ANN, GPU Mini-Batch or speedup claim is included.
The probe overlaps training data; small movement under large cumulative counts
and missed rare clusters can still cause premature heuristic stopping. Neither
probe stability nor final exact evaluation proves full-data convergence. Tune
policies on development data before any held-out evaluation; compare quality,
processed observations and full runtime rather than equal update counts alone.

## Reproducible integration workflow (October 2026)

The integrated code is developed on `integration-daksh`. `requirements-lock.txt`
records the entire tested Python 3.12.3 environment, including NumPy 2.2.6,
Numba 0.65.1, numba-cuda 0.30.4, HNSW 0.8.0, Rasterio 1.5.1,
Matplotlib 3.11.2 and pandas 3.1.0. On a **new** environment, after approving
installation, use `python3.12 -m venv .venv` then
`.venv/bin/python -m pip install -r requirements-lock.txt` and
`.venv/bin/python -m pip check`. HNSW may need a C++ compiler and Python 3.12
headers to build; do not change system packages or an existing validated
installation automatically. Rasterio wheels bundle GDAL; no system GDAL install
is required. Use the WSL NVIDIA driver bridge; never install Linux GPU drivers
inside WSL. The lock records this environment, not universal platform support.

Algorithm execution in the integrated runner is explicit:

| ID | Implementation | Execution |
|---|---|---|
| C0 | Full Lloyd reference | Sequential CPU |
| C1 | Fixed Mini-Batch, exact assignment | Sequential CPU, explicit opt-in |
| G0 | Custom exact full Lloyd | CUDA GPU |
| G1 | Fixed Mini-Batch, exact assignment | CUDA GPU |
| G2 | Fixed Mini-Batch + HNSW | CPU prototype |
| G3 | Adaptive batch, exact assignment | CPU prototype |
| G4 | Fixed batch + adaptive HNSW effort | CPU prototype |
| G5 | Joint adaptive batch/HNSW effort | CPU prototype |
| G3CUDA | Adaptive batch, exact assignment | CUDA GPU, explicit opt-in |
| G2CUDA | Fixed batch + projection-window ANN | CUDA GPU, explicit opt-in |
| G4CUDA | Fixed batch + adaptive projection effort | CUDA GPU, explicit opt-in |
| G5CUDA | Adaptive batch + projection effort | CUDA GPU, explicit opt-in |
| O0 | Full Lloyd, C++ OpenMP | CPU, explicit opt-in |

G3CUDA calls `pcap.adaptive_cuda.fit_adaptive_minibatch_cuda`. It preserves
GPU-resident cumulative sums/counts, centroids and input points across size
changes; only sampled indices, monitoring data and final results cross the bus.
Buffers are sized once for the maximum batch, with active slices. It uses the
same controller/checkpoint/confirmation rules as the CPU adaptive baseline,
including three fresh frozen maximum-batch confirmation checks. It rejects CUDA
simulation/unavailable devices, and requires float64-atomic support. Host timings
include synchronization and transfers; no isolated per-kernel timing or speedup
is inferred. G2/G4/G5 remain CPU HNSW prototypes. G2CUDA/G4CUDA/G5CUDA use a
separate custom sorted-projection candidate index with GPU distance evaluation;
the ANN index itself is built on the CPU. Candidate effort is not HNSW `ef`.
The bounded RTX 4050 pilot found this projection method slower end-to-end than
exact GPU assignment at the tested N=8192 settings. At low effort its recall
and objective gap were poor, while small K and effort>=K used explicitly exact
search. It is a feasibility result, not a performance win. See
[`docs/IMPLEMENTATION_PLAN_STATUS.md`](docs/IMPLEMENTATION_PLAN_STATUS.md).

O0 compiles `native/kmeans_openmp.cpp` using the installed `g++` and the explicit
`scripts/build_openmp.sh` command; the runner never compiles code implicitly.
The real-crop pilot used the same X/C0 and 55 Lloyd updates at 1, 2, 4 and 8
threads. Four threads were fastest in that single session; eight threads were
slower than four. This limited run does not establish general CPU scaling.

### Real imagery: required external input, never synthesized

Raw SAFE products are user-provided and gitignored; none are bundled in the
repository. Fixture JP2 filenames contain synthetic GTiff payloads and must
never be presented as genuine SAFE evidence. `data/independent_scene_candidates.json`
records the acquired validation and test products and their checksums. Their
512×512 crops use the frozen development statistics. Exact K=16, seed=0 CPU/GPU
reference fits and maps exist for the 31UFU validation crop and 30STG test crop.
The latest all-method held-out K=16 map set is under
`outputs/20261025-test-scene-all-method-maps/`; for example,
`render_G0/true_color.png`, `render_G0/false_color_nir.png`,
`render_G0/clusters.png` and `render_G0/clusters.tif` show the exact CUDA result.
The folder also contains the corresponding C0 and approximate/adaptive method
maps, per-method checksums and provenance. Earlier exact-only maps remain under
`outputs/20261025-test-scene-k16-exact-cap1000/` and
`outputs/20261025-validation-scene-k16-exact/`. These output folders are local,
gitignored artifacts and must be preserved/backed up separately from source.
The
test crop has not been used to select or tune adaptive settings; do not use its
results for tuning. A frozen K=16 test-quality evaluation now covers five seeds
and one policy: exact C0/G0 reached the declared 5% target in 25/25 fits per
execution group; Mini-Batch, ANN and adaptive variants missed it by the
1,000-step cap. CPU and CUDA method groups ran separately, so these results do
not support cross-device speedup claims. See `data/README.md` and
`docs/REAL_SCENE_RESULTS.md` for the protocol, per-method results and limits.
The five-seed summary is reproducible from the saved run archives with
`experiments/summarize_holdout_quality.py`; its current output is
`outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/`.

Example after those files exist (replace paths and product directory names):

```bash
# splits.json maps each input product directory to train/validation/test.
.venv/bin/python -B scripts/prepare_sentinel2_scenes.py \
  --raw-dir /path/to/approved/SAFE-products --output-dir outputs/prepared-NEW \
  --training-scene-id TRAIN_PRODUCT.SAFE --split-manifest /path/to/splits.json \
  --crop-size 512 --row 2000 --column 2000

MPLCONFIGDIR=/tmp/pcap-mplconfig bash scripts/run_suite_cuda.sh \
  --scene-dir outputs/prepared-NEW/TRAIN_PRODUCT.SAFE \
  --methods C0 G0 G1 G2 G3 G4 G5 G3CUDA --k 4 16 --seeds 0 1 2 \
  --batch-size 256 --max-steps 30 --warmups 2 --repeats 5 \
  --budget-seconds 120 --save-models --output-dir outputs/real-study-NEW

MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B experiments/plot_results.py \
  --csv outputs/real-study-NEW/benchmark_results_TIMESTAMP.csv \
  --output-dir outputs/real-study-NEW/figures
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B experiments/summarize_suite.py \
  --csv outputs/real-study-NEW/benchmark_results_TIMESTAMP.csv \
  --output-dir outputs/real-study-NEW/scaling
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B scripts/render_scene.py \
  --scene-dir outputs/prepared-NEW/TRAIN_PRODUCT.SAFE \
  --model-dir outputs/real-study-NEW/models_TIMESTAMP/nN_k4_seed0_G0_rep0 \
  --output-dir outputs/real-study-NEW/scene-map-NEW
```

A 120-second budget is checked **between** complete fits and includes runner
work; use `timeout 150s` for a process limit when appropriate. An outer kill can
prevent final manifest writing, so the internal limit must be smaller. Failed
fits/budgets propagate exceptions and save failures/partial measurements; missing
scenes and hash mismatches fail explicitly. Raw JSONL includes warmups, histories,
internal timings and hashes; warmups are excluded from measured CSV/statistics.
The runner rotates method order, retains outliers, and does not establish
fixed-work equivalence across different algorithms. G0/G1/G3CUDA require actual
GPU access; no fallback occurs. Output directories must be new for preparation
and rendering. Timestamped model/results files and SHA256 inventories preserve
individual runs; no previous experiment directory should be reused for figures.

Models save exact final labels/centroids and JSON histories. Rendering checks
scene/model hashes and exact assignments, outputs georeferenced `clusters.tif`
with nodata=-1, and preserves masked/unsampled pixel positions. Display-only
2–98% stretches produce B04/B03/B02 true-color and B08/B04/B03 false-color PNGs;
these do not change features. Histories produce adaptive batch/effort plots.
These are cluster IDs, never independently verified land-cover classes.

The summary utility keeps N/D/K/seed separate and reports median/IQR, individual
run CSVs, SSE, observations and convergence/cap status. Exact C0/G0 speedups are
reported only for matching steps/termination and compatible final SSE; they are
external full-fit ratios. Mini-Batch comparisons require matched quality or
quality-versus-time curves, not iteration counts. `experiments/benchmark_quality.py`
provides a separate instrumented time-to-target study: it freezes a target from
a converged exact CPU reference, performs the same full-data quality checkpoints
inside each measured fit, and reports capped runs as target misses. Do not pool
these instrumented timings with the legacy runner timings. Real-data quality
coverage includes a five-seed validation K sweep and a five-seed held-out K=16
study, but it is not the full multi-scene N/K matrix. Do not pool capped and
converged protocols or claim a CPU ANN prototype is a GPU speedup. The OpenMP
baseline now exists; instructor confirmation that Python-authored CUDA
satisfies the proposal's C/C++ language requirement remains outstanding.

### New evaluation pilots (25 October 2026)

The final-source projection-ANN feasibility archive is
`outputs/20261025-projection-feasibility-final/`. It contains 392 raw queries
(112 warmups and 280 measured) over N=8192, D=4/14 and K=4..256, with two
warmups per case and five measurements, raw timings, recall/objective gaps, source snapshots, hardware
conditions and a verified recursive SHA256 inventory. Exact GPU assignment was
faster end-to-end for all approximate settings in this pilot. The observed
negative result is recorded; no ANN speedup is claimed.

The real-data quality study is `outputs/20261025-real-quality-256-k16/`:
65,536 pixels from the existing genuine 31UFS product, D=4, K=16, seed=0,
two warmups and five measured fits for C0/G0/G1/G2CUDA/G3CUDA/G4CUDA/G5CUDA.
Its frozen target was 19,583.2966 SSE (5% over the converged exact reference).
C0 and G0 reached it in all five repeats; the Mini-Batch, ANN and adaptive GPU
variants reached the 30-step cap without reaching it. See the CSV, raw histories
and target plot in that directory. This is one quality slice, not a complete
ablation or held-out result.

The real-data OpenMP archive is `outputs/20261025-openmp-real-maastricht-256/`.
It contains five measured fits at each requested thread count, 1/2/4/8, two
warmups, identical inputs and exact outputs (55 steps). Median external O0 fit
times were 83.57, 54.74, 44.35 and 72.58 ms, respectively; relative to C0 in
each session these are 1.01x, 1.54x, 1.88x and 1.20x. The 8-thread result shows
why one short session cannot establish a stable scaling curve.

Reproduce the explicit OpenMP build and runner call with a newly prepared scene:

```bash
bash scripts/build_openmp.sh
.venv/bin/python -B experiments/run_suite.py \
  --scene-dir outputs/prepared-NEW/TRAIN_PRODUCT.SAFE \
  --methods C0 O0 --omp-threads 4 --k 4 --seeds 0 \
  --max-steps 200 --warmups 2 --repeats 5 \
  --output-dir outputs/openmp-real-NEW
```

Reproduce the quality-study command after a real prepared crop exists:

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig timeout 150s bash scripts/run_cuda_python.sh \
  experiments/benchmark_quality.py \
  --scene-dir outputs/prepared-NEW/TRAIN_PRODUCT.SAFE \
  --output-dir outputs/quality-NEW --methods C0 G0 G1 G2CUDA G3CUDA G4CUDA G5CUDA \
  --k 16 --seeds 0 --policy-json configs/default_policies.json --batch-size 512 \
  --max-steps 30 --reference-steps 1000 --evaluation-interval 5 \
  --warmups 2 --repeats 5 --budget-seconds 120
.venv/bin/python -B experiments/plot_quality.py \
  --study outputs/quality-NEW --output-dir outputs/quality-NEW/figures
```

Final regression commands (CUDA failures are errors, not skips):

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B -m pytest tests -q -p no:cacheprovider \
  --ignore=tests/test_cuda_kmeans.py --ignore=tests/test_cuda_minibatch.py \
  --ignore=tests/test_cuda_vector_add.py --ignore=tests/test_cuda_adaptive.py \
  --ignore=tests/test_cuda_ann.py
MPLCONFIGDIR=/tmp/pcap-mplconfig bash scripts/run_cuda_tests.sh tests -x
```

## Genuine satellite validation

Three Sentinel-2 L2A products (development 31UFS, validation 31UFU and test
30STG) have been processed from genuine SAFE/XML/JP2 files. Exact CPU/CUDA,
Mini-Batch, ANN and adaptive variants ran on real crops and generated
georeferenced maps. See [real-scene details, artifacts and reproduction](docs/REAL_SCENE_RESULTS.md).
The frozen policy misses its held-out quality target for Mini-Batch/ANN/adaptive
variants, and beneficial GPU ANN remains unsupported by the available results.

Additional validation covers the independent 31UFU scene at K=4/8/16/32 with
five seeds, and the frozen 30STG test protocol covers all twelve methods at
K=16 (five seeds). Only exact C0/G0 reached the test scene's 5% target (25/25
fits per execution group); the other methods hit the 1,000-step cap. All twelve
methods have saved aligned test cluster maps and same-sample unsupervised
metrics. A separate one-update
fixed-work study uses genuine nested 31UFS windows with N=1,048,548 and
4,194,265, D=4/K=16, five seeds and five measured repetitions. Its paired exact
C0/G0 CPU/GPU ratios were 1.087× and 2.052× in those two sessions; these are
single-update observations, not convergence or adaptive speedups. See
[`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md) for full tables,
limitations, archive paths, and commands to reproduce the summary plots.

The completed real-scene time-to-frozen-quality matrix covers valid
N=262,144 on the validation scene and N=1,048,548/4,194,265 on the development
scene, at K=4/8/16/32 with five seeds.
At N≥1,048,548 the quality runs used a 100-update cap; many approximate and
adaptive methods missed the fixed 5% target, and some exact K=16/32 runs also
hit the cap. These target misses are retained, not relabeled as convergence or
speedups. The held-out 30STG quality report remains frozen at K=16; no test-scene
results were used to retune parameters. See the matrix table and per-run archive
in [`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md).

### Latest verified image and report figures

These are existing, checksum-inventoried local artifacts (not tracked Git
files). The main visual comparison is
`outputs/20261024-real-evaluation/real-scene-comparison.png`. The all-method
held-out cluster maps and RGB/NIR views are in
`outputs/20261025-test-scene-all-method-maps/render_<METHOD>/` (including
`clusters.tif`, `clusters.png`, `true_color.png` and `false_color_nir.png`).
For the latest aggregate quality results, use
`outputs/20261026-real-quality-matrix-completion/quality_target_hit_rate.png`
with its `manifest.json` and `individual_runs.csv`. The validation K-sweep
speedup figure is
`outputs/20261025-validation-quality-k-sweep-summary-v2/exact_speedup_by_k.png`;
the five-seed held-out quality-versus-time figures are
`outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/quality_seed{0..4}_{cpu,cuda}.png`.
Their protocols differ; do not pool or compare those plots as if they were one
measurement session. Full interpretation and limitations are in
[`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md).

A separate paired validation-only diagnostic compared the unchanged cumulative-
count Mini-Batch rule with an experimental decayed-count update on identical
31UFU K=16 data, initialization, batch-index stream and 5,000-update budget.
The cumulative run ended at SSE 28,276.14 without reaching the frozen target
25,614.79; the 10-pass-half-life decayed run first crossed it at checkpoint
2,100 and ended at 24,703.84. This single result supports cumulative-history
inertia as a cause for that validation case, but does not establish a generally
better update rule. The production cumulative-count baseline is unchanged; the
decayed variant remains experimental and was not evaluated on the untouched
test scene. Exact methodology, one-run timings, checkpoint metrics, reproduction
command and verified artifact checksums are in
[`docs/REAL_SCENE_RESULTS.md`](docs/REAL_SCENE_RESULTS.md). The plan audit and
prioritized final-submission gates are in
[`docs/IMPLEMENTATION_PLAN_STATUS.md`](docs/IMPLEMENTATION_PLAN_STATUS.md).

The plan requests kernel-level profiling support, but the WSL Nsight Systems
installation did not produce a usable trace. Aligned ground-truth land-cover
labels were not available; cluster IDs are unsupervised and no land-cover
accuracy is claimed. The instructor scope question for Python-authored CUDA
versus C/C++ is prepared in [`docs/INSTRUCTOR_CLARIFICATION.md`](docs/INSTRUCTOR_CLARIFICATION.md).

## References and project resources

The following references and official resources are starting points from the
project plan. Cite a source in the final report only for claims it directly
supports.

- Pant et al. (2025), project-plan background reference: [DOI 10.3390/a18080532](https://doi.org/10.3390/a18080532).
- Han and Lee (2024), project-plan background reference: [DOI 10.3390/rs16010102](https://doi.org/10.3390/rs16010102).
- Li, Frank, and Pfahringer (2023), project-plan background reference: [DOI 10.1007/s10618-022-00869-6](https://doi.org/10.1007/s10618-022-00869-6).
- Spalding-Jamieson et al. (2025), project-plan background reference: [arXiv:2502.06163](https://arxiv.org/abs/2502.06163).
- Official implementation/data resources: [Numba-CUDA documentation](https://nvidia.github.io/numba-cuda/), [FAISS K-Means documentation](https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering), and [Copernicus Browser](https://browser.dataspace.copernicus.eu/).

The instructor scope question remains unanswered; see
[`docs/INSTRUCTOR_CLARIFICATION.md`](docs/INSTRUCTOR_CLARIFICATION.md). The
current gate-by-gate status and items needed before final submission are in
[`docs/IMPLEMENTATION_PLAN_STATUS.md`](docs/IMPLEMENTATION_PLAN_STATUS.md).
