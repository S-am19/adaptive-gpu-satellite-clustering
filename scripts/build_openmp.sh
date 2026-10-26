#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/outputs/build"
g++ -O3 -std=c++17 -fopenmp -ffp-contract=off "$root/native/kmeans_openmp.cpp" -o "$root/outputs/build/pcap_openmp"
g++ --version | head -n 1
"$root/.venv/bin/python" -B - "$root" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);exe=root/'outputs/build/pcap_openmp'
info=dict(compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
          flags='-O3 -std=c++17 -fopenmp -ffp-contract=off',
          source_sha256=hashlib.sha256((root/'native/kmeans_openmp.cpp').read_bytes()).hexdigest(),
          executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
Path(str(exe)+'.json').write_text(json.dumps(info,indent=2))
PY
