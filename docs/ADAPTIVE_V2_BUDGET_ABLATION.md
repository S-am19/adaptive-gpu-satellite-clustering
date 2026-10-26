# Adaptive V2 observation-budget ablation

## Scope and protocol

This is a single validation-scene experiment, not a timing replication. It uses
the 31UFU validation crop (`N=262144`, `D=4`), `K=16`, seeds 0, 1 and 2,
identical per-seed `X`, `C0`, and `FitConfig` within each method comparison,
batch 8192 initially, maximum batch 65536, tolerance `1e-6`, and a 40,960,000
processed-observation cap for fixed decayed and Adaptive V2. Full-data SSE was
evaluated at shared 8,192,000-observation milestones. Adaptive batches can
cross a milestone by at most one batch; the last batch was shortened when
needed to end exactly at the shared cap. Exact CUDA G0 ran to its normal
convergence criterion.

The frozen 5% targets were evaluation-only and never passed to any fit or
controller. They were 25614.786371 (seed 0), 23841.053797 (seed 1), and
25668.873196 (seed 2). Each method used its seeded uniform-without-replacement
sampler. The per-update SHA256 of sampled point indices and the complete update
history are archived. Fixed and adaptive runs use the same RNG seed, but their
sampled index sequences diverge after their batch widths differ; they are not a
point-for-point common sample stream.

## Results

| Seed | Method | End-to-end fit time | First observed target time | Final exact SSE | Gap to exact reference |
|---:|---|---:|---:|---:|---:|
| 0 | Exact CUDA G0 | 1.443 s | 1.287 s | 24395.034639 | approximately 0% |
| 0 | Fixed decayed | 26.580 s | 13.702 s | 24703.840421 | 1.266% |
| 0 | Adaptive V2 | 13.076 s | 10.054 s | 24703.775636 | 1.266% |
| 1 | Exact CUDA G0 | 1.422 s | 1.208 s | 22705.765521 | approximately 0% |
| 1 | Fixed decayed | 31.012 s | not reached | 24167.063332 | 6.436% |
| 1 | Adaptive V2 | 12.773 s | not reached | 24167.570821 | 6.438% |
| 2 | Exact CUDA G0 | 1.474 s | 1.085 s | 24446.545901 | approximately 0% |
| 2 | Fixed decayed | 23.459 s | 23.441 s | 25429.699827 | 4.022% |
| 2 | Adaptive V2 | 9.330 s | 9.306 s | 25435.953699 | 4.047% |

At equal total observations, Adaptive V2 and fixed decayed had nearly the same
quality. The final SSE differences (V2 minus fixed) were `-0.065`, `+0.507`,
and `+6.254` for seeds 0, 1, and 2. Across the five shared milestones, the
sign of the small SSE difference varied by seed and checkpoint. This does not
show a consistent quality advantage for adaptive batch sizing.

V2 completed the same observation budget in 13.076, 12.773, and 9.330 seconds,
versus 26.580, 31.012, and 23.459 seconds for fixed decayed in this one run.
That is an observed 2.03x, 2.43x, and 2.51x elapsed-time difference. The V2
update counts were 1649, 1528, and 903, compared with 5000 fixed updates. The
larger batches reduce per-update and launch/monitoring overhead. This is
promising evidence of a throughput advantage over fixed decayed Mini-Batch,
but each method/seed has only one timed fit, and runs are subject to GPU clock
and scheduling variation.

Exact CUDA G0 reached the same frozen target in all three seeds, at 1.085–1.287
seconds. Both decayed methods missed seed 1; in seeds 0 and 2, their observed
crossings took 9.306–23.441 seconds. Neither Mini-Batch method approached G0's
time-to-target in this experiment.

## Controller history

All nine controller batch increases (three per seed) were triggered by
`probe_objective_stagnation`: each of three consecutive probe improvements was
below the policy's 0.1% stagnation threshold. For example, seed 0 increased
8192 to 16384 at update 170 after a 0.0874% probe improvement; it later
increased to 32768 and 65536 after 0.0810% and 0.0752% improvements. The other
seeds showed the same sub-threshold pattern. No deterioration or oscillation
triggered these increases. Shrinking requires sustained improvement above
0.5% plus low normalized movement and label changes, which was not observed at
the growth checkpoints. At maximum batch size, clipped stagnation requests
were logged as no-ops. The final shortened batches were budget truncations,
not controller decisions.

The history also records probe SSE, normalized centroid movement, label-change
fraction, decision reason and states, adaptation/confirmation status, and a
SHA256 digest for each sampled batch. Target metrics are kept separate from
these controller records.

All six Mini-Batch runs ended exactly at the 40.96M observation cap without
satisfying the convergence criterion. The original run metadata labeled these
stops `max_steps`; the recovered summary records the effective stop reason as
`observation_budget` and preserves the original raw label alongside it.

## Limitations and interpretation

- This is one run per method and seed, on one validation crop. The timing ratios
  are exploratory and should not be presented as replicated speedups.
- Fixed and adaptive sampling use the same per-run seed but not identical
  point-index sequences after batch-size schedules diverge. Per-batch hashes
  and sampler provenance are retained so the actual draws can be audited.
- The comparison uses identical observation caps, not identical update counts.
- The quality target is a clustering-objective criterion, not land-cover
  accuracy; the scene has no labeled ground truth for that claim.
- The run completed all nine fits and exact final label/SSE checks. Its original
  CSV summary writer failed on a method-specific timing column. The original
  failure manifest and partial CSV are preserved; `measurements_reconstructed.csv`
  was regenerated from complete raw records without rerunning fits. See
  `recovery.json` and `checksums.sha256` in the archive.
- The archived outputs are ignored local artifacts at
  `outputs/20261026-adaptive-v2-observation-budget-40m/`; they are not included
  in Git by this document alone.

## Conclusion for reporting

The prior matched fixed-cumulative versus fixed-decayed runs support decayed
counts as a useful quality improvement over cumulative updates at the tested
work budget, subject to the sampling-provenance caveat above. This ablation
does not show that adaptive batch sizing improves final quality at equal work.
It does show a potentially substantial wall-time reduction versus fixed
decayed Mini-Batch in one three-seed run, with similar final SSE, while exact
CUDA remains much faster to the same target. Present the adaptive timing result
as preliminary; replicate timings before making a performance claim. Retain
V2 as a labeled experimental method and close further tuning on the held-out
test scene.
