#!/usr/bin/env bash
# Configure driver discovery only for this test process and its children.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="$project_root/.venv/bin/python"
if [[ ! -x "$python_executable" ]]; then
    echo "Project Python not found: $python_executable" >&2
    exit 1
fi

kernel_release="$(uname -r)"
if [[ "${kernel_release,,}" == *microsoft* && "${kernel_release,,}" == *wsl2* ]]; then
    wsl_driver_dir=/usr/lib/wsl/lib
    if [[ ! -r "$wsl_driver_dir/libcuda.so" ]]; then
        echo "WSL2 CUDA driver bridge missing: $wsl_driver_dir/libcuda.so" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$wsl_driver_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export NUMBA_ENABLE_CUDASIM=0
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_DISABLE=1
cd -- "$project_root"
if (( $# == 0 )); then
    set -- tests/test_cuda_vector_add.py
fi
exec "$python_executable" -B -m pytest -q -s -p no:cacheprovider "$@"
