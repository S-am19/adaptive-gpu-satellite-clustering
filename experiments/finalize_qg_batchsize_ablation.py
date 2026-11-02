"""Finalize already measured QGCUDA batch-size ablation records (no reruns)."""
import csv
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/20261027-qg-batchsize-ablation"
csv.field_size_limit(2**31 - 1)

from run_qg_batchsize_ablation import aggregate, make_plots, write_csv, METHODS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path, json_fields=()):
    with Path(path).open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    for row in records:
        for field in json_fields:
            if row.get(field):
                row[field] = json.loads(row[field])
    return records


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Verify the completed runner's own archive before deriving any new files.
    checksum_path = OUT / "checksums.sha256"
    if not checksum_path.is_file():
        raise RuntimeError("Missing run-time checksums.sha256")
    checksum_failures = []
    for line in checksum_path.read_text().splitlines():
        expected_hash, relative = line.split("  ", 1)
        if not (OUT / relative).is_file() or sha(OUT / relative) != expected_hash:
            checksum_failures.append(relative)
    if checksum_failures:
        raise RuntimeError(f"Run archive checksum verification failed before finalization: {checksum_failures}")
    config = json.loads((OUT / "config.json").read_text())
    runs = read_records(OUT / "individual_runs.csv",
                        ("timings", "metadata", "batch_schedule", "randomized_order"))
    checkpoints = read_records(OUT / "checkpoints.csv")
    events = read_records(OUT / "controller_events.csv",
                          ("controller_decision", "controller_state_before", "controller_state_after"))
    expected = len(config["seeds"]) * config["measured_repetitions"] * len(METHODS)
    if len(runs) != expected or expected != 105:
        raise RuntimeError(f"Expected all 105 measured fits before finalization; got {len(runs)}")
    if {r["method"] for r in runs} != set(METHODS):
        raise RuntimeError("One or more declared methods are missing")
    keys = [(int(r["seed"]), int(r["repeat"]), r["method"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate seed/repetition/method rows found")
    if any(r["X_sha256"] != config["x_sha256"] for r in runs):
        raise RuntimeError("Feature matrix hash mismatch")
    for seed in config["seeds"]:
        for repeat in range(config["measured_repetitions"]):
            pair = [r for r in runs if int(r["seed"]) == seed and int(r["repeat"]) == repeat]
            if len(pair) != len(METHODS) or len({r["C0_sha256"] for r in pair}) != 1:
                raise RuntimeError(f"Methods do not share inputs/initialization for seed={seed}, repeat={repeat}")
            for row in pair:
                if row["method"].startswith("FIXED_") or row["method"] in ("QGCUDA", "G3CUDA"):
                    if int(row["processed_observations"]) != config["observation_budget"]:
                        raise RuntimeError("Mini-Batch fit did not complete the common observation budget")
    # Preserve a hash record for the raw records and source as they existed
    # before finalization; finalization writes only derived summaries/figures.
    raw_hashes = {name: sha(OUT / name) for name in (
        "individual_runs.csv", "checkpoints.csv", "controller_events.csv",
        "timing_breakdown.csv", "reference_runs.csv", "warmups.csv", "config.json")}
    summary = aggregate(runs)
    write_csv(OUT / "method_summary.csv", summary)

    paired = []
    for adaptive in ("QGCUDA", "G3CUDA"):
        for fixed in ("FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536"):
            by_fixed = {(int(r["seed"]), int(r["repeat"])): r for r in runs if r["method"] == fixed}
            by_adapt = {(int(r["seed"]), int(r["repeat"])): r for r in runs if r["method"] == adaptive}
            for key in sorted(by_fixed):
                f, a = by_fixed[key], by_adapt[key]
                paired.append(dict(seed=key[0], repeat=key[1], fixed_method=fixed,
                    adaptive_method=adaptive, same_X_hash=f["X_sha256"] == a["X_sha256"],
                    same_C0_hash=f["C0_sha256"] == a["C0_sha256"],
                    same_processed_observations=int(f["processed_observations"]) == int(a["processed_observations"]),
                    fixed_wall_s=float(f["external_wall_s"]), adaptive_wall_s=float(a["external_wall_s"]),
                    adaptive_over_fixed_wall_ratio=float(a["external_wall_s"])/float(f["external_wall_s"]),
                    fixed_sse=float(f["final_exact_sse"]), adaptive_sse=float(a["final_exact_sse"]),
                    adaptive_minus_fixed_sse=float(a["final_exact_sse"])-float(f["final_exact_sse"]),
                    fixed_target_hit=f["target_hit"], adaptive_target_hit=a["target_hit"],
                    both_hit_target=f["target_hit"] == "True" and a["target_hit"] == "True",
                    fixed_steps=int(f["steps"]), adaptive_steps=int(a["steps"]),
                    comparison="paired complete-fit; target-quality comparison only if both hit frozen target"))
    write_csv(OUT / "paired_comparisons.csv", paired)

    make_plots(OUT, runs, checkpoints)
    fixed_summary = [s for s in summary if s["method"].startswith("FIXED_")]
    best_fixed = min(fixed_summary, key=lambda r: r["median_external_wall_s"])
    qg = next(s for s in summary if s["method"] == "QGCUDA")
    g3 = next(s for s in summary if s["method"] == "G3CUDA")
    best_fixed_runs = [r for r in runs if r["method"] == best_fixed["method"]]
    qg_runs = [r for r in runs if r["method"] == "QGCUDA"]
    best_pairs = [r for r in paired if r["adaptive_method"] == "QGCUDA" and
                  r["fixed_method"] == best_fixed["method"]]
    qg_win_count = sum(r["adaptive_over_fixed_wall_ratio"] < 1 for r in best_pairs)
    qg_tie_count = sum(np.isclose(r["adaptive_over_fixed_wall_ratio"], 1.0) for r in best_pairs)
    qg_target_matches = sum(r["both_hit_target"] for r in best_pairs)
    qg_median_delta = statistics.median(r["adaptive_minus_fixed_sse"] for r in best_pairs)
    report = f"""# Fixed batch-size versus adaptive CUDA Mini-Batch ablation

## Protocol

Real Sentinel-2 31UFU validation crop, N={config['n']:,}, D={config['d']}, K={config['k']}; seeds {', '.join(map(str, config['seeds']))}; three measured repetitions per method. The four fixed CUDA arms use batches 8,192, 16,384, 32,768, and 65,536. They are compared with existing QGCUDA, existing G3CUDA, and exact CUDA G0. Each Mini-Batch arm uses cumulative float64 sums/int64 counts, exact squared-Euclidean assignment, the same seed-derived 1,024-point probe withheld from training, five common exact full-data SSE checkpoints, and exactly {config['observation_budget']:,} attempted observations. All methods for a seed/repetition share the same C0. Batch sizes change the number and partition of RNG draws; seeded without-replacement sampling policy is common, but sampled index sequences are not pointwise identical.

The 5% target is derived from the seed-specific exact CPU reference and is evaluation-only. Neither adaptive controller receives the target. G0 is exact CUDA Lloyd; it does not run a Mini-Batch observation budget and is included as the exact quality/time reference. Complete fit wall time from `perf_counter` is primary. Internal categories include probe monitoring, controller, and full-data quality evaluation, and are not summed to reconstruct wall time.

## Results

| Method | Median complete wall (s) | IQR (s) | Median exact SSE | Median SSE gap | Target hits | Median steps | Median processed observations |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {s['method']} | {s['median_external_wall_s']:.3f} | {s['wall_iqr_s']:.3f} | {s['median_final_exact_sse']:.2f} | {s['median_sse_gap_to_exact_reference']:.4%} | {s['target_hits']}/{s['n_runs']} | {s['median_steps']:.0f} | {s['median_processed_observations']:.0f} |"
        for s in summary) + f"""

The fastest fixed arm by median wall time was **{best_fixed['method']}** ({best_fixed['median_external_wall_s']:.3f}s; {best_fixed['target_hits']}/{best_fixed['n_runs']} target hits). QGCUDA was {qg['median_external_wall_s']:.3f}s ({qg['target_hits']}/{qg['n_runs']} hits), and G3CUDA was {g3['median_external_wall_s']:.3f}s ({g3['target_hits']}/{g3['n_runs']} hits). In the paired QGCUDA-versus-{best_fixed['method']} comparisons, QGCUDA was faster in {qg_win_count}/{len(best_pairs)} pairs (ties: {qg_tie_count}); median SSE difference (QGCUDA minus fixed) was {qg_median_delta:.2f}. Both reached the frozen target in {qg_target_matches}/{len(best_pairs)} pairs.

The hypothesis that QGCUDA beats the best fixed batch is assessed against the best fixed arm selected by **median runtime across the predeclared five-seed, three-repeat matrix**, not against only the 8,192 baseline. If the target-hit counts differ, runtime alone does not establish a quality-equivalent advantage. See `paired_comparisons.csv` for all paired deltas and `method_summary.csv` for medians/IQRs. `checkpoints.csv`, `controller_events.csv`, `timing_breakdown.csv`, `individual_runs.csv`, and `warmups.csv` retain the underlying observations, schedules, probe/controller costs, and individual timings.

## Interpretation limits

This is a single validation acquisition, one K, and one observation budget. The best fixed arm is selected within this experiment and therefore should be treated as a validation result, not a universally optimal batch size. Different batch widths yield different seeded sample partitions. The quality target was not a controller input. No CPU/GPU speedup is inferred from this CUDA-only ablation. These records do not establish land-cover classification accuracy.
"""
    (OUT / "final_summary.md").write_text(report, encoding="utf-8")
    (OUT / "README.md").write_text(f"""# QGCUDA fixed-batch-size ablation

This is a real-GPU experiment on Sentinel-2 31UFU validation data (N={config['n']}, D={config['d']}, K={config['k']}). It compares fixed CUDA batch sizes 8192, 16384, 32768, and 65536 against QGCUDA, G3CUDA, and exact CUDA G0. See `final_summary.md` for interpretation and `manifest.json`/`config.json` for provenance and protocol.

## Reproduction

Run from the repository root with the existing `.venv` and real CUDA GPU. The command requires a new output directory and preserves the original archive:

```bash
bash scripts/run_cuda_python.sh experiments/run_qg_batchsize_ablation.py \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --reference-dir outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/full_disjoint_probe \\
  --output-dir outputs/20261027-qg-batchsize-ablation-reproduction \\
  --seeds 0 1 2 3 4 --repeats 3 --warmups 2 \\
  --observation-budget 40960000 --evaluation-interval 8192000 \\
  --k 16 --max-steps 12000 --time-budget-seconds 3900
```

The frozen quality target is evaluator-only. Timing components overlap; use `external_wall_s` as the complete-fit timing. Each run's sampling policy uses a seeded generator and a common disjoint probe, but batch widths produce different sampled-index sequences.
""", encoding="utf-8")

    finalization = dict(finalized_utc=datetime.now(timezone.utc).isoformat(),
        measured_rows=len(runs), expected_rows=expected,
        input_archive_checksums_verified=True,
        original_postprocessing_failure=manifest.get("failures", []),
        interpretation="All 105 fits were recorded; initial run failed only while generating the batch-trajectory plot. This finalizer produced derived outputs without rerunning fits.",
        runner_sha256_at_experiment=manifest.get("source_hashes", {}).get("experiments/run_qg_batchsize_ablation.py"),
        corrected_plotter_sha256=sha(ROOT / "experiments/run_qg_batchsize_ablation.py"),
        finalizer_sha256=sha(Path(__file__)), raw_hashes_before_finalization=raw_hashes,
        best_fixed_method=best_fixed["method"], qgcuda_paired_wins_over_best_fixed=qg_win_count,
        qgcuda_paired_comparisons=len(best_pairs))
    (OUT / "finalization_provenance.json").write_text(json.dumps(finalization, indent=2))
    finalizer_snapshot = OUT / "source/experiments/finalize_qg_batchsize_ablation.py"
    finalizer_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), finalizer_snapshot)
    manifest["complete"] = True
    manifest["execution_complete"] = True
    manifest["measured_fits_expected"] = expected
    manifest["measured_fits_recorded"] = len(runs)
    manifest["failures"] = []
    manifest["postprocessing_failures_preserved"] = [
        "TypeError: batch-schedule plot initially parsed an in-memory list as JSON; all measured rows had completed and been saved"]
    manifest["finalization_provenance"] = "finalization_provenance.json"
    manifest["finalized_utc"] = finalization["finalized_utc"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
    checksum = OUT / "checksums.sha256"
    checksum.write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in sorted(files)))
    bad = []
    for line in checksum.read_text().splitlines():
        expected_hash, relative = line.split("  ", 1)
        if sha(OUT / relative) != expected_hash:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Checksum verification failed: {bad}")
    print(f"Finalized {len(runs)} recorded fits; verified {len(files)} output files at {OUT}")


if __name__ == "__main__":
    main()
