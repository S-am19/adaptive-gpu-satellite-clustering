"""M0 smoke test: require a real CUDA GPU and verify a custom kernel."""

from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
from numba import cuda


@cuda.jit
def vector_add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


def test_vector_add_on_cuda_gpu():
    assert not cuda.config.ENABLE_CUDASIM, "CUDA simulator must be disabled"
    cuda_module = Path(cuda.__file__).resolve()
    assert "numba_cuda" in cuda_module.parts, cuda_module
    assert cuda.is_available(), "A real CUDA GPU is required"

    # Device 0 is the first GPU visible through CUDA_VISIBLE_DEVICES.
    cuda.select_device(0)
    device = cuda.get_current_device()
    print(f"Python: {sys.version.split()[0]}; executable: {sys.executable}")
    print(f"CUDA module: {cuda_module}")
    print(f"GPU: {device.name}; compute capability: {device.compute_capability}")
    print(f"CUDA simulator: {cuda.config.ENABLE_CUDASIM}")
    for package in ("numpy", "numba", "numba-cuda", "pytest", "llvmlite"):
        print(f"{package}: {version(package)}")

    n = 65_539  # Includes a partial final block to exercise the bounds check.
    a = np.arange(n, dtype=np.float32)
    b = np.full(n, 0.5, dtype=np.float32)
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.to_device(np.full(n, np.nan, dtype=np.float32))
    threads = 256
    blocks = (n + threads - 1) // threads
    vector_add[blocks, threads](d_a, d_b, d_out)
    cuda.synchronize()
    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result, a + b)
    assert vector_add.signatures, "Kernel must have a compiled specialization"
    print(f"Verified {n} float32 additions; grid={blocks}, block={threads}")
