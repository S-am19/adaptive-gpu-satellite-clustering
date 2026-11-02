# Historical integration validation snapshot — 24 October 2026

> **Historical snapshot, superseded for current status.** This file records the
> state before real SAFE products were acquired and before the later CUDA,
> OpenMP, real-scene and quality-matrix work. Its statements below are retained
> as a dated record and must not be read as current facts. For current status,
> see [`IMPLEMENTATION_PLAN_STATUS.md`](IMPLEMENTATION_PLAN_STATUS.md) and the
> later [`REAL_SCENE_RESULTS.md`](REAL_SCENE_RESULTS.md).

## Acceptance status

**Not complete for final project acceptance.** No genuine Sentinel-2 SAFE product
was found in the searched project/WSL project folders or Windows user
Downloads/Documents/Desktop/OneDrive folders. No imagery was downloaded, no
credentialed service was accessed, and no real-scene map or held-out quality
claim is made. Synthetic/georeferenced fixtures validate plumbing, not real
ESA XML/JP2 integration. `data/scene_candidates.json` is public OData metadata
queried on 24 October 2026, not an acquired dataset or cloud-quality selection.

The three listed candidate products total 3,011,588,508 compressed bytes (~3.01 GB,
2.80 GiB). Acquiring them requires explicit download approval and potentially
Copernicus authentication supplied securely outside source/chat logs; alternatively
provide already acquired SAFE paths. Cloud suitability, crop coverage and spatial
independence still need checking. Do not tune on held-out scenes. Catalogue:
https://catalogue.dataspace.copernicus.eu/odata/v1/Products ; access documentation:
https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html .

## Implemented and validated

- Custom exact CPU/CUDA Lloyd baselines; fixed CPU/CUDA Mini-Batch;
  CPU HNSW Mini-Batch, adaptive ANN effort and combined adaptive CPU controls.
- Additional opt-in G3CUDA exact adaptive Mini-Batch with persistent GPU data,
  cumulative float64 sums/int64 counts, real controller transitions and frozen
  maximum-batch confirmation. G2/G4/G5 remain explicitly CPU prototypes.
- Source/dependency provenance, bounded pilot runner, warmup exclusion, exact
  final label/SSE gates, explicit failures and partial artifacts, saved models,
  median/IQR summaries and matched-work exact speedup validation.
- Crop offset controls, no preprocessing overwrite, train-only standardization,
  hash-checked model/scene rendering, aligned GeoTIFF cluster IDs/nodata,
  true/NIR display composites, adaptation histories and SHA256 inventories.

## Measurements (synthetic only)

The two experiments are separate protocols. All measurements include complete
fit calls, validation, allocation/transfer, monitoring, ANN maintenance and final
exact evaluation within the implementation's timing scope. Dataset generation,
quality metrics, verification and artifact writing are outside the external fit
timer. Histories and internal overlapping diagnostic timings are preserved;
never add those fields into totals. No isolated kernel profiling is claimed.
GPU: actual RTX 4050 Laptop, 6141 MiB total; initial free 5921 MiB, P8, 41 C,
210 MHz, 0% utilization. This snapshot does not establish causation for variation.

### Short all-method pipeline pilot

`outputs/20261024-integration-pilot/`: N=256/1024, D=4, K=4/16, seeds 0/1,
batch=64, max_steps=10, tolerance=1e-5, warmups=2, repeats=5, rotating order.
320 measured fits + 128 warmups; no failed fits. Exact C0/G0 matched-work speedups
(CPU median / GPU median) ranged **0.0504–0.4148x**: CPU was faster in all eight
cases. No controller changed effort or batch size in this short protocol. All
adaptive runs were capped; this is not an adaptive-benefit or convergence study.

### Longer adaptation diagnostic

`outputs/20261024-adaptation-diagnostic/`: N=1024, D=4, K=4, seeds 0/1,
batch=512, max_steps=100, tolerance=1e-5, warmups=2, repeats=5. C1/G1/G2/G3/G4/G5/
G3CUDA; 70 measured fits + 28 warmups, no failed fits. All capped at 100 updates.
Seed 0 made no changes; seed 1 changed batch/effort in the relevant adaptive
variants. Times below are medians; individual times/IQR are in archived CSVs.

| Seed | Method | Median full fit (ms) | Final exact SSE | Observations |
|---|---|---:|---:|---:|
| 0 | C1 fixed CPU | 10.581 | 6410.501909 | 51200 |
| 0 | G3 adaptive CPU batch | 11.840 | 6410.501909 | 51200 |
| 0 | G2 fixed CPU ANN | 71.676 | 6410.501909 | 51200 |
| 0 | G4 adaptive CPU ANN | 82.811 | 6410.501909 | 51200 |
| 0 | G5 combined CPU | 75.600 | 6410.501909 | 51200 |
| 1 | C1 fixed CPU | 12.823 | 1370.703402 | 51200 |
| 1 | G3 adaptive CPU batch | 11.788 | 1370.720330 | 48640 |
| 1 | G2 fixed CPU ANN | 75.798 | 1370.703402 | 51200 |
| 1 | G4 adaptive CPU ANN | 78.508 | 1370.703402 | 51200 |
| 1 | G5 combined CPU | 77.253 | 1370.720330 | 48640 |

Seed-1 batch adaptation reduced observations by 5% and took ~1.088x less median
CPU time, at a relative SSE increase of ~0.00124%. Seed 0 was slower. ANN effort
adaptation did not beat fixed ANN on either seed. Small K=4 HNSW overhead was
substantial. These are descriptive capped-fit results, not matched target-quality
speedups, land-cover accuracy or held-out evidence. GPU fixed/adaptive medians
were 120.401/96.419 ms (seed 0), 556.329/577.968 ms (seed 1); adaptive seed-1 IQR
was 581.744 ms. This variability prevents a stable acceleration conclusion.

Original run manifests, raw JSONL, model arrays/JSON and summary CSVs are
checksummed. Source snapshots were verified against each manifest's hashes;
the short pilot's runner was reconstructed by removing the later C1 dispatch
addition and its SHA256 verified. Original measurements were never rewritten.
Derived plot/source inventories are recorded separately. These ignored local
outputs must be transferred separately when handing off the project.

## Remaining work and scientific limits

1. Approved acquisition and genuine SAFE/JP2 preprocessing, real cluster maps,
   multiple spatially independent scenes and held-out evaluation.
2. GPU ANN/joint adaptation is not implemented. HNSW is a CPU prototype; the
   project must retain this limitation or approve a defensible GPU ANN work stage.
3. Time-to-target-quality evaluation, more seeds/sessions and larger real-data
   scaling are needed. Current final-quality scatter is not a convergence curve.
4. Probe stability with cumulative averages is heuristic and can miss rare
   clusters. Exact final labels/SSE evaluate the returned centroids, not global
   optimality. ANN build/query/rebuild costs are included; no evaluation-count
   reduction is claimed without measurements.
5. Profiler tooling remains blocked; do not interpret grouped events as isolated
   kernel timings. No profiling tools were installed or investigated further.
6. Original OpenMP and Python-vs-CUDA-C++ requirements need instructor resolution;
   no silent requirement change or claim of completed original scope is made.

## Final verification and reproduction

Final observed validation: **343 CPU-only tests passed** (82 warnings),
**393 real-GPU tests passed** (101 warnings), with RTX 4050 and CUDA simulator 0.
Warnings are small-grid performance hints and pending Matplotlib/Rasterio/Affine
deprecations. `pip check`, Python source parsing and `git diff --check` passed.
No shared-contract changes, package installations, imagery downloads, commits,
pushes or merges were performed in this implementation phase. All prior
integration-daksh edits remain present. This working tree is not a completed
real-data project and is not ready for final acceptance/merge as such.

Reproduce the synthetic pilot in a new output directory:

```bash
MPLCONFIGDIR=/tmp/pcap-mplconfig timeout 150s bash scripts/run_suite_cuda.sh \
  --methods C0 G0 G1 G2 G3 G4 G5 G3CUDA --sizes 256 1024 --d 4 --k 4 16 \
  --seeds 0 1 --batch-size 64 --max-steps 10 --warmups 2 --repeats 5 \
  --budget-seconds 120 --save-models --output-dir outputs/pilot-REPRO-NEW
MPLCONFIGDIR=/tmp/pcap-mplconfig timeout 150s bash scripts/run_suite_cuda.sh \
  --methods C1 G1 G2 G3 G4 G5 G3CUDA --sizes 1024 --d 4 --k 4 --seeds 0 1 \
  --batch-size 512 --max-steps 100 --warmups 2 --repeats 5 \
  --budget-seconds 120 --save-models --output-dir outputs/adaptation-REPRO-NEW
```

Use the generated timestamped CSV with `experiments/summarize_suite.py` and
`experiments/plot_results.py`; add `--raw-jsonl PATH` to the latter for histories.
See the README for final test commands, SAFE preprocessing and geospatial render
commands. For exact historical source provenance use each saved `source/` tree;
its bytes were verified against that run's original manifest. No pooling of
sessions, seeds or protocols is justified. Output inventories can be verified
from their directory with `sha256sum -c checksums_TIMESTAMP.sha256` and
`sha256sum -c final-derived-checksums.sha256`.
