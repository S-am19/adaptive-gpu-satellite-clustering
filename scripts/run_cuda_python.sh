#!/usr/bin/env bash
# Run a project Python CLI with process-local WSL driver discovery.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# == 0 )); then echo 'Usage: bash scripts/run_cuda_python.sh SCRIPT [arguments...]' >&2; exit 2; fi
release="$(uname -r)"
if [[ "${release,,}" == *microsoft* && "${release,,}" == *wsl2* ]]; then
    if [[ ! -r /usr/lib/wsl/lib/libcuda.so ]]; then echo 'WSL CUDA driver bridge unavailable' >&2; exit 1; fi
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export NUMBA_ENABLE_CUDASIM=0 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pcap-mplconfig}"
cd -- "$root"
exec "$root/.venv/bin/python" -B "$@"
