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
