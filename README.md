# adaptive-gpu-satellite-clustering
GPU-accelerated adaptive Mini-Batch K-Means for satellite image clustering using Python and CUDA

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
tests that pass NumPy arrays. Real SAFE/XML/JP2 integration remains a separate
validation step; these tests do not download satellite data.

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

G3CUDA calls `pcap.adaptive_cuda.fit_adaptive_minibatch_cuda`. It preserves
GPU-resident cumulative sums/counts, centroids and input points across size
changes; only sampled indices, monitoring data and final results cross the bus.
Buffers are sized once for the maximum batch, with active slices. It uses the
same controller/checkpoint/confirmation rules as the CPU adaptive baseline,
including three fresh frozen maximum-batch confirmation checks. It rejects CUDA
simulation/unavailable devices, and requires float64-atomic support. Host timings
include synchronization and transfers; no isolated per-kernel timing or speedup
is inferred. G2/G4/G5 have **not** become GPU ANN implementations. A beneficial
GPU ANN technique at small K remains an unresolved project objective.

### Real imagery: required external input, never synthesized

No genuine Sentinel-2 imagery is bundled, downloaded or validated by these
commands alone. Fixture JP2 filenames contain synthetic GTiff payloads and must
never be presented as genuine SAFE evidence. `data/scene_candidates.json` contains
public-catalogue candidates, not acquired/quality-accepted scenes. Obtain approval
before large downloads or credentialed service access. Supply original L2A SAFE
products with product-root XML and matching 10 m B02/B03/B04/B08 and 20 m SCL.
Select nonoverlapping development/validation/test scenes and inspect cloud masks;
freeze policy settings before held-out evaluation. See `data/README.md`.

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
quality-versus-time curves, not iteration counts. The current runner does not
implement time-to-target trajectories; its final-quality scatter is not a
convergence curve. Do not pool capped and converged protocols or claim a CPU
ANN prototype is a GPU speedup. Large-K ANN feasibility and held-out real-scene
performance still need measured evidence. OpenMP and instructor approval of the
Python/CUDA scope remain outstanding.

Final regression commands (CUDA failures are errors, not skips):

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig .venv/bin/python -B -m pytest tests -q -p no:cacheprovider \
  --ignore=tests/test_cuda_kmeans.py --ignore=tests/test_cuda_minibatch.py \
  --ignore=tests/test_cuda_vector_add.py --ignore=tests/test_cuda_adaptive.py
MPLCONFIGDIR=/tmp/pcap-mplconfig bash scripts/run_cuda_tests.sh tests -x
```
