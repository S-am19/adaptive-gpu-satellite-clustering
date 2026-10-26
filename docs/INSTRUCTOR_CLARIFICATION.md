# Instructor clarification

Our project has a sequential Python/Numba CPU implementation, a native C++
OpenMP CPU baseline, and custom CUDA kernels authored in Python with
Numba-CUDA and executed on the NVIDIA GPU. The proposal names C/C++ and custom
CUDA.

Could you confirm whether Python-authored Numba-CUDA kernels satisfy the
proposal's CUDA C/C++ requirement? If CUDA C++ source is required, should we
port the GPU kernels, or is the native C++ OpenMP baseline sufficient for the
CPU-parallel component while the GPU kernels remain Python-authored?
