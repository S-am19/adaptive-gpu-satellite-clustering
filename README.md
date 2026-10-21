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
