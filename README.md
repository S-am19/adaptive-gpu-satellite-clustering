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
