#!/usr/bin/env bash
# Process-local WSL driver discovery; no CUDA simulation or CPU fallback.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
release="$(uname -r)"
if [[ "${release,,}" == *microsoft* && "${release,,}" == *wsl2* ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export NUMBA_ENABLE_CUDASIM=0
export PYTHONDONTWRITEBYTECODE=1
exec "$root/.venv/bin/python" -B "$root/experiments/run_suite.py" "$@"
