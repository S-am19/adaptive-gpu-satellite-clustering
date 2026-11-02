# Implementation plan status — audited 26 October 2026

This is an evidence-based status against `IMPLEMENTATION_PLAN.md`. “Implemented”
describes code and tests; it does not imply that every experimental or
instructor-approval gate is complete.

| Stage | Status | Evidence and remaining gate |
|---|---|---|
| M0 — environment | Implemented on Samridhi's WSL2 system | Python 3.12.3, pinned venv, real RTX 4050 CUDA vector-add and regression runs. Daksh's separate WSL installation has not been checked from this machine. |
| M1 — sequential CPU oracle | Implemented | Seeded exact Lloyd CPU, shared contracts, exact final SSE and CPU tests. |
| M2 — exact CUDA | Implemented | Custom Numba CUDA assignment/accumulation kernels, GPU-resident inputs, synchronization, CPU agreement and real-GPU tests. No speedup claim for all workloads. |
| M3 — Sentinel-2 preprocessing | Implemented on three genuine products | B02/B03/B04/B08, product XML offsets/quantification, georeferenced SCL nearest-neighbor mask, CRS checks, training-only normalization, pixel mapping, and exact CPU/GPU maps for development, validation and test crops. |
| M4 — fixed Mini-Batch | Implemented | CPU/CUDA variants preserve cumulative statistics, exact final labels/SSE and stopping metadata. |
| M5 — ANN | Implemented; CUDA feasibility negative | CPU HNSW and custom CUDA projection-window candidate assignment exist. In the synthetic RTX 4050 pilot, approximate CUDA search was slower end-to-end than exact GPU assignment over the tested range; ANN GPU acceleration is not established. |
| M6 — adaptation | Implemented; measured benefit not established | Batch-only, CPU ANN-effort, CUDA projection-effort and combined controllers have tests and runners. Validation and test studies show misses at the stated caps; controller operation does not establish better time-to-quality. |
| M7 — evaluation | Core real-image matrix measured; profiling and repeated-session coverage remain incomplete | Five-seed validation quality sweep at N=262,144, K=4/8/16/32; five-seed development quality matrix at N=1,048,548 and 4,194,265, K=4/8/16/32; frozen five-seed held-out K=16 evaluation; five-seed one-update scaling at N≈1.05M/4.19M. Full raw records, target misses, source provenance and checksums are preserved. Large-N quality runs cap at 100 updates and have one/two measured fits per seed in one session. A controlled validation K=16 decayed-count diagnostic now reaches the frozen target once; it is a one-scene experimental alternative, not a replacement policy. Nsight did not yield per-kernel traces. |
| M8 — proposal scope | Open pending instructor response | Native C++ OpenMP and a 1/2/4/8-thread pilot exist. Instructor confirmation that Python-authored CUDA satisfies the proposal's C/C++ requirement is still needed. |

## New validation and scaling evidence

`outputs/20261025-validation-quality-k-sweep-summary-v2/` summarizes four
separate 31UFU validation studies on the same 512×512 prepared crop
(N=262,144, D=4), each with five seeds, five measured fits per method, two
warm-ups, max_steps=100, full-data checks every 10 updates, and the frozen
policy. Each study's raw CSV/JSONL, source snapshot, target trajectories and
recursive checksums remain in its own directory:

- `outputs/20261025-validation-quality-k4-frozen-policy/`
- `outputs/20261025-validation-quality-k8-frozen-policy/`
- `outputs/20261025-validation-quality-k16-frozen-policy-100/`
- `outputs/20261025-validation-quality-k32-frozen-policy/`

Exact C0/G0 reached the 5%-over-reference target in all 25 measured runs at
each K. At K=4, the other methods reached the target in 15/25 runs; at K=8, 16,
and 32, none reached it by the 100-update cap. `quality_target` is not
tolerance convergence. Exact paired C0/G0 median speedups (CPU time / CUDA
time) were 0.744×, 0.930×, 1.363× and 3.283× for K=4/8/16/32; their IQRs were
0.341×, 0.204×, 0.211× and 1.536×. The summarizer validates matching X/C0
hashes, updates, termination, target status and final SSE before reporting
these ratios. Each K is its own one-session validation workload; these results
do not prove a universal speedup or adaptive advantage.

The fixed-work real-image scaling studies use nested Maastricht windows from
the genuine 31UFS SAFE product, frozen Maastricht512 training statistics,
D=4/K=16, five seeds, two warm-ups and five measured runs per method:

- N=1,048,548 valid pixels: `outputs/20261025-real-fixed-work-N1048548-K16/`
- N=4,194,265 valid pixels: `outputs/20261025-real-fixed-work-N4194265-K16/`
- Charts and paired summaries: `outputs/20261025-real-fixed-work-scaling-summary-v2/`

Every fit performs one update and is marked `max_steps`; adaptive controllers
cannot adapt in this one-step protocol. For exact C0/G0, paired median
CPU/GPU ratios were 1.087× (IQR 0.090×) at N≈1.05M and 2.052× (IQR 0.696×)
at N≈4.19M. These are single-session fixed-work observations. Mini-Batch and
ANN methods process different assignment work, so their times are not
presented as hardware speedups.

The frozen held-out 30STG test quality study covers one K=16 policy, five seeds,
and five measured repeats per method in separate CPU and CUDA groups. Exact
C0/G0 reached the seed-specific 5% target in 25/25 fits per execution group;
every Mini-Batch/ANN/adaptive method missed it by `max_steps=1,000`. Test data
was not used to retune settings. All twelve methods also have saved,
exact-label-validated cluster rasters with verified CRS/transform under
`outputs/20261025-test-scene-all-method-maps/`. Five-seed summaries and plots
are preserved under `outputs/20261025-test-quality-k16-five-seed-summary-rebuilt/`.

## Remaining acceptance limits

1. The primary real-image N/K quality matrix is covered for five seeds on the
   development scene at N=262,144/1,048,548/4,194,265 and K=4/8/16/32, with the
   262,144 validation-scene K sweep also complete. The held-out test remains
   frozen at K=16 (five seeds); it was not used for tuning. D=14 remains
   synthetic-only because the real workflow uses four specified 10 m bands.
2. At the 100-update large-N cap, many Mini-Batch/ANN/adaptive fits and some
   exact K=16/32 fits did not meet the frozen 5% target. These are explicit
   target misses, not convergence. The original cumulative-count fixed and
   adaptive K=16 validation runs through 5,000 updates still missed. A single
   matched decayed-count diagnostic on that validation crop crossed the same
   target at a 2,100-update checkpoint; this supports cumulative-history
   inertia as a cause for that case, but does not establish generalization or
   justify replacing the frozen baseline. No correctness-test defect was
   identified. The default adaptive controller reduced batch size to its
   minimum in the 5,000-update run, reducing observations processed.
3. A complete five-seed, five-repeat held-out K=4/8/32 sweep was not run.
   Additional held-out K sweeps are not needed to tune and are intentionally
   not used to alter frozen parameters. A supplemental interrupted K=4 CUDA
   run is archived but excluded from conclusions.
4. Independent repeated measurement sessions for each large-N/method case
   remain outstanding; the completed large-N matrix uses one session and one
   or two measured fits per seed. Synthetic D=14 scaling exists separately.
5. Nsight Systems 2022.4.2 did not produce a usable CUDA timeline because its
   importer was unavailable. Kernel-level profiling is called for by the M7
   measurements plan but remains an environmental limitation; no per-kernel
   bottleneck claims are supported.
6. The plan makes external land-cover comparison optional when aligned labels
   are available. No labeled reference raster was supplied, so no semantic
   accuracy is claimed; cluster IDs remain unsupervised.
7. Instructor confirmation of Python-authored Numba CUDA versus the proposal's
   C/C++ wording remains necessary before final scope acceptance. See
   `docs/INSTRUCTOR_CLARIFICATION.md`.

The completed test scene's artifacts and source snapshots are local, ignored
outputs, not Git deliverables. Regenerate them with the documented commands;
do not commit raw SAFE data, `.venv`, or result arrays.

## Final-submission checklist

### Plan gates still requiring a decision or evidence

1. **Instructor language/scope decision (blocking for strict M8 acceptance):**
   obtain a written answer to `docs/INSTRUCTOR_CLARIFICATION.md` on whether
   Python-authored Numba-CUDA satisfies the proposal's C/C++ requirement. If
   CUDA C++ is required, porting and validating the kernels is still work; the
   existing C++ OpenMP code does not automatically satisfy a CUDA-language
   requirement.
2. **Kernel profiling (plan measurement still outstanding):** Section 8 asks
   for kernel, synchronization and transfer measurements/profile diagnostics.
   The attempted Nsight Systems capture produced no usable trace. Grouped CUDA
   event totals are not per-kernel measurements. Either obtain a usable profiler
   trace without mislabeling timings or get instructor acceptance of this
   documented limitation.
3. **Second developer environment (M0 exit evidence):** M0 specifies real-GPU
   smoke validation on both WSL installations. Samridhi's RTX 4050 is verified;
   Daksh's own WSL/Python/CUDA setup is not evidenced in this repository.
4. **Git publication (not yet authorized):** at this audit, the local
   `integration-daksh` branch was clean at `5275aa1922dda97debde3f183e2d00e451f8c4a3`,
   five commits ahead of its configured `origin/feature-daksh` tracking ref;
   no `origin/integration-daksh` ref was present. Local `main` and
   `samridhi-dev` were at `2e8856dc16eac4f326b28d95d73a8214219dacea`.
   These branch observations are a dated snapshot, not a live status check.
   Publishing the integration branch and any later merge to `main` require
   explicit authorization. Keep ignored datasets/results backed up separately.

### Experiment coverage: complete, partial, and optional

- **Sufficient for the report:** real development/validation quality coverage
  at N=262,144, 1,048,548 and 4,194,265 for K=4/8/16/32; five seeds per
  matrix cell (large-N cells have one or two measured fits per seed); a frozen
  five-seed held-out K=16 study; exact C0/G0 matched runs; fixed Mini-Batch,
  ANN, batch-only, ANN-only and joint method results; one-update scaling;
  real maps and source-linked artifacts. Quality misses and caps are valid
  negative outcomes and must remain visible.
- **Incomplete under a strict all-splits reading:** the held-out 30STG scene
  has a complete K=16 five-seed study, but not complete K=4/8/32 studies. The
  supplemental K=4 CUDA run was interrupted and is excluded. The plan does not
  unambiguously require repeating every K on the held-out scene; ask the
  instructor before expanding it. Do not tune using those held-out results.
- **Replication limitation:** the large-N quality matrix is one session with
  one or two measured fits per seed, rather than multiple sessions. The plan
  requires repeated seeds and spread, which exist, but broad independent
  session replication remains an improvement rather than a demonstrated
  acceptance gate.
- **Not required by the initial spectral scope:** real imagery uses the four
  native 10 m bands (D=4); D=14 remains synthetic-only. A 4096 crop is
  explicitly optional after memory checks. Higher-K ANN feasibility is covered
  synthetically; no real high-K sweep is claimed.
- **Optional if reference labels are available:** no aligned land-cover truth
  raster was supplied. Silhouette/DBI and cluster maps are unsupervised; no
  land-cover accuracy is claimed. This is a documented limitation, not a
  blocker under the plan's optional-label wording.
- **ANN feasibility:** the custom CUDA projection candidate search has a
  measured negative feasibility result in its tested range; the plan permits a
  documented go/no-go finding. It must not be described as a GPU ANN speedup.

### Documentation and handoff tasks

- Keep `docs/REAL_SCENE_RESULTS.md` as the source of detailed protocols,
  per-case outcomes, limitations, archive paths and reproduction commands;
  keep README claims linked to those artifacts and avoid pooling protocols.
- Include the decayed-count result as one validation diagnostic only. The
  cumulative-count `fit_minibatch_cpu`/CUDA implementations remain the named
  fixed Mini-Batch baselines; the decayed rule is separately labeled and is
  not part of the frozen held-out evaluation.
- At the 26 October audit, the working tree was clean and all source changes
  were committed locally; the CPU (380) and real-GPU (450) suites had passed,
  and archived-output checksums were verified. Re-run those checks if source
  changes after that audited commit. The instructor scope decision and kernel
  profiling acceptance remain open, so do not describe the project as accepted
  under the original proposal or merged into `main`.
