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
