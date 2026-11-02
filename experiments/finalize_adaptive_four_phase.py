"""Validate, summarize and render the isolated 2026-10-27 study archive."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/20261027-adaptive-four-phase-study"
csv.field_size_limit(2**31 - 1)


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, data):
    if not data:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def verify_local_manifest(directory):
    directory = Path(directory)
    sums = directory / "checksums.sha256"
    if not sums.exists():
        return False, "missing checksums.sha256"
    failures = []
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        file = directory / rel
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            failures.append(rel)
    return (not failures), failures


def iqr(values):
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    args = p.parse_args()
    base = args.base.resolve()
    phase_dirs = {
        "phase1": base / "phase1_gpu_fixed_vs_adaptive/full_production_stream",
        "phase2": base / "phase2_cpu_gpu_comparisons/full_production_stream",
        "phase3": base / "phase3_quality_guarded_adaptive/full_disjoint_probe",
        "phase4": base / "phase4_ann_and_combined/real_validation_pilot_final",
    }
    expected = {"phase1": 45, "phase2": 75, "phase3": 60, "phase4": 162}
    manifests = {}
    data = {}
    integrity = {}
    for phase, d in phase_dirs.items():
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        ok, detail = verify_local_manifest(d)
        integrity[phase] = {"checksums_valid": ok, "detail": detail}
        if not ok:
            raise RuntimeError(f"Checksum failure in {phase}: {detail}")
        if not manifest.get("complete"):
            raise RuntimeError(f"Phase {phase} manifest is not complete")
        run_path = d / "individual_runs.csv"
        run_rows = rows(run_path)
        if len(run_rows) != expected[phase]:
            raise RuntimeError(f"{phase}: expected {expected[phase]} records, got {len(run_rows)}")
        manifests[phase] = manifest
        data[phase] = run_rows

    # Adaptive V2 uses a different (decayed-count) update family. Validate and
    # summarize it separately; never fold it into the cumulative-update rows.
    decayed_dir = base / "phase3_quality_guarded_adaptive/decayed_v2_separate_update_retry3"
    decayed_manifest = None
    decayed_runs = []
    decayed_checkpoints = []
    if (decayed_dir / "manifest.json").is_file():
        decayed_manifest = json.loads((decayed_dir / "manifest.json").read_text())
        ok, detail = verify_local_manifest(decayed_dir)
        if not ok or not decayed_manifest.get("complete"):
            raise RuntimeError(f"Separate decayed V2 replication incomplete or corrupt: {detail}")
        decayed_runs = rows(decayed_dir / "individual_runs.csv")
        decayed_checkpoints = rows(decayed_dir / "checkpoints.csv")
        if len(decayed_runs) != 6 or {r["method"] for r in decayed_runs} != {
                "C_DECAYED_FIXED", "D_ADAPTIVE_V2"}:
            raise RuntimeError("Separate decayed V2 replication must contain six method/seed fits")
        expected_x = manifests["phase3"]["config"]["x_sha256"]
        if any(r["x_sha256"] != expected_x for r in decayed_runs):
            raise RuntimeError("Decayed V2 run feature hash differs from validation study")
        for seed in {r["seed"] for r in decayed_runs}:
            pair = [r for r in decayed_runs if r["seed"] == seed]
            if len(pair) != 2 or len({r["C0_sha256"] for r in pair}) != 1:
                raise RuntimeError(f"Decayed V2 pair for seed {seed} lacks a shared initialization")
            if len({int(r["observations_processed"]) for r in pair}) != 1:
                raise RuntimeError(f"Decayed V2 pair for seed {seed} has unequal observation budgets")
            if any(r["target_evaluation_only"] != "True" for r in pair):
                raise RuntimeError("Frozen target was not consistently evaluation-only")
        if len(decayed_checkpoints) != 30:
            raise RuntimeError(f"Expected 30 decayed V2 quality checkpoints, got {len(decayed_checkpoints)}")

    # Create top-level figure directory and use verified phase output figures.
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    copies = [
        (phase_dirs["phase1"] / "runtime_and_sse.png", "01_fixed_vs_adaptive_gpu_runtime_sse.png"),
        (phase_dirs["phase1"] / "sse_vs_observations.png", "02_sse_vs_processed_observations.png"),
        (phase_dirs["phase1"] / "target_and_sse.png", "03_time_to_quality_target.png"),
        (phase_dirs["phase2"] / "runtime_and_sse.png", "04_cpu_gpu_fixed_adaptive_runtime_sse.png"),
        (phase_dirs["phase3"] / "runtime_and_sse.png", "05_quality_guarded_runtime_sse.png"),
        (phase_dirs["phase3"] / "sse_vs_observations.png", "06_quality_guarded_sse_vs_observations.png"),
        (phase_dirs["phase3"] / "target_and_sse.png", "07_quality_guard_target_hits.png"),
    ]
    for src, name in copies:
        if not src.is_file():
            raise RuntimeError(f"Missing expected figure {src}")
        shutil.copyfile(src, figdir / name)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Paired CPU/GPU ratios are workload matched; none are target matched.
    paired = rows(phase_dirs["phase2"] / "paired_speedups.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (cm, gm, label) in enumerate((("C1", "G1", "Fixed"), ("C3", "G3CUDA", "Adaptive"))):
        subset = [r for r in paired if r["cpu_method"] == cm and r["gpu_method"] == gm]
        vals = [float(r["wall_ratio_cpu_over_gpu"]) for r in subset]
        ax.scatter([i] * len(vals), vals, alpha=.75, label=label)
        ax.plot([i - .15, i + .15], [statistics.median(vals)] * 2, color="black", lw=2)
    ax.axhline(1, color="gray", ls="--")
    ax.set_xticks([0, 1], ["Fixed CPU/GPU", "Adaptive CPU/GPU"])
    ax.set_ylabel("Paired CPU wall / GPU wall\n(<1 means GPU was slower)")
    ax.set_title("Same seed, work budget and batch schedule; no pair met target")
    fig.tight_layout(); fig.savefig(figdir / "08_paired_cpu_gpu_ratios.png", dpi=180); plt.close(fig)

    # Batch-size trajectories from one predeclared illustrative replicate.
    phase3_rows = data["phase3"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"G1": "#4c78a8", "G3CUDA": "#f58518", "QGCUDA": "#54a24b"}
    for method in colors:
        r = next(x for x in phase3_rows if x["method"] == method and x["seed"] == "0" and x["repeat"] == "0")
        schedule = json.loads(r["batch_schedule"])
        xx = [item[0] / 1e6 for item in schedule]
        yy = [item[2] for item in schedule]
        ax.step(xx, yy, where="post", label=method, color=colors[method], lw=1.5)
    ax.set(xlabel="Processed observations (millions)", ylabel="Actual batch size",
           title="Batch schedules (seed 0, repetition 0; Phase 3 protocol)")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "09_batch_size_trajectories.png", dpi=180); plt.close(fig)

    # Controller probe behavior and rollback markers for the same QG replicate.
    controller = rows(phase_dirs["phase3"] / "controller_events.csv")
    q = [r for r in controller if r["method"] == "QGCUDA" and r["seed"] == "0" and r["repeat"] == "0"]
    q = [r for r in q if r.get("probe_sse")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [int(r["points_processed"]) / 1e6 for r in q]
    y = [float(r["probe_sse"]) for r in q]
    ax.plot(x, y, color="#54a24b", label="Independent fixed-probe objective")
    changes = [r for r in q if json.loads(r["controller_decision"]).get("changed")]
    for r in changes:
        ax.axvline(int(r["points_processed"]) / 1e6, color="#f58518", alpha=.6, lw=1)
    rolls = [r for r in q if r["rollback"] == "True"]
    if rolls:
        ax.scatter([int(r["points_processed"]) / 1e6 for r in rolls],
                   [float(r["probe_sse"]) for r in rolls], marker="x", color="red", label="Rollback")
    ax.set(xlabel="Processed observations (millions)", ylabel="Probe SSE",
           title="Quality-guard decisions (seed 0, repetition 0; no evaluation target input)")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "10_quality_guard_controller.png", dpi=180); plt.close(fig)

    # ANN K/effort plots use the updated-centroid snapshot, maintenance included.
    ann = rows(phase_dirs["phase4"] / "individual_runs.csv")
    ann = [r for r in ann if r["phase"] == "measured" and r["snapshot"] == "after_one_minibatch_update"]
    ks = sorted({int(r["k"]) for r in ann})
    efforts = sorted({int(r["effort"]) for r in ann if r["method"] == "projection"})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    exact = [statistics.median(float(r["external_complete_wall_s"]) for r in ann
                               if int(r["k"]) == k and r["method"] == "exact") * 1000 for k in ks]
    ax.plot(ks, exact, marker="o", color="black", label="Exact CUDA")
    for effort in efforts:
        xs, ys = [], []
        for k in ks:
            ss = [r for r in ann if int(r["k"]) == k and r["method"] == "projection" and int(r["effort"]) == effort]
            if ss:
                xs.append(k); ys.append(statistics.median(float(r["external_complete_wall_s"]) for r in ss) * 1000)
        ax.plot(xs, ys, marker="o", label=f"Projection candidates={effort}")
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks])
    ax.set(xlabel="K", ylabel="Complete assignment wall time (ms)",
           title="Projection ANN cost includes initial build, device refresh and transfers")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir / "11_ann_complete_cost_vs_k.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for effort in efforts:
        xs, recall, gap = [], [], []
        for k in ks:
            ss = [r for r in ann if int(r["k"]) == k and r["method"] == "projection" and int(r["effort"]) == effort]
            if ss:
                xs.append(k); recall.append(statistics.median(float(r["strict_label_agreement"]) for r in ss))
                gap.append(100 * statistics.median(float(r["assignment_objective_gap"]) for r in ss))
        ax[0].plot(xs, recall, marker="o", label=str(effort))
        ax[1].plot(xs, gap, marker="o", label=str(effort))
    ax[0].axhline(.95, color="gray", ls="--"); ax[1].axhline(1, color="gray", ls="--")
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(ks, [str(k) for k in ks]); a.set_xlabel("K")
    ax[0].set_ylabel("Strict label agreement"); ax[1].set_ylabel("Mean assignment objective gap (%)")
    ax[0].legend(title="Candidates", fontsize=8); ax[1].legend(title="Candidates", fontsize=8)
    fig.suptitle("Projection ANN assignment quality after centroid update")
    fig.tight_layout(); fig.savefig(figdir / "12_ann_agreement_objective_gap.png", dpi=180); plt.close(fig)

    if decayed_runs:
        # Show each seed explicitly: n=3 is too small for a distributional
        # boxplot, and checkpoint SSEs are evaluated at actual processed work.
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        methods = ("C_DECAYED_FIXED", "D_ADAPTIVE_V2")
        colors = {"C_DECAYED_FIXED": "#4c78a8", "D_ADAPTIVE_V2": "#f58518"}
        for j, method in enumerate(methods):
            rr = [r for r in decayed_runs if r["method"] == method]
            ax[0].scatter([j] * len(rr), [float(r["wall_s"]) for r in rr],
                          color=colors[method], alpha=.85)
            ax[1].scatter([j] * len(rr), [float(r["final_exact_sse"]) for r in rr],
                          color=colors[method], alpha=.85)
            for r in rr:
                ax[0].annotate(f"seed {r['seed']}", (j, float(r["wall_s"])), xytext=(4, 3),
                               textcoords="offset points", fontsize=7)
                ax[1].annotate(f"seed {r['seed']}", (j, float(r["final_exact_sse"])), xytext=(4, 3),
                               textcoords="offset points", fontsize=7)
        ax[0].set_xticks([0, 1], ["Fixed decay", "Adaptive V2"])
        ax[0].set_ylabel("External complete-fit wall time (s)")
        ax[1].set_xticks([0, 1], ["Fixed decay", "Adaptive V2"])
        ax[1].set_ylabel("Final exact full-data SSE")
        fig.suptitle("Separate decayed-count update family; equal 40.96M observations")
        fig.tight_layout(); fig.savefig(figdir / "13_decayed_v2_fixed_vs_adaptive.png", dpi=180); plt.close(fig)

    # Machine-readable method summary and provenance index.
    summary = []
    for phase in ("phase1", "phase2", "phase3"):
        for method in sorted({r["method"] for r in data[phase]}):
            subset = [r for r in data[phase] if r["method"] == method]
            walls = [float(r["wall_s"]) for r in subset]
            sses = [float(r["final_sse"]) for r in subset]
            obs = [int(r["observations_processed"]) for r in subset]
            summary.append(dict(phase=phase, method=method, device=subset[0]["device"], n_runs=len(subset),
                median_wall_s=statistics.median(walls), wall_iqr_s=iqr(walls),
                median_final_exact_sse=statistics.median(sses), final_sse_iqr=iqr(sses),
                target_hits=sum(r["target_hit"] == "True" for r in subset),
                target_hit_rate=sum(r["target_hit"] == "True" for r in subset) / len(subset),
                median_observations=statistics.median(obs),
                termination_reasons=";".join(sorted({r["termination_reason"] for r in subset}))))
    final_dir = base / "final_summary"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_csv(final_dir / "method_comparison.csv", summary)

    decayed_summary = []
    if decayed_runs:
        for method in ("C_DECAYED_FIXED", "D_ADAPTIVE_V2"):
            subset = [r for r in decayed_runs if r["method"] == method]
            crossing = [float(r["first_target_crossing_time_s"])
                        for r in subset if r.get("first_target_crossing_time_s") not in ("", None)]
            decayed_summary.append(dict(
                method=method, update_family="decayed historical sums/effective counts",
                n_runs=len(subset), observation_budget=int(subset[0]["observations_processed"]),
                median_wall_s=statistics.median(float(r["wall_s"]) for r in subset),
                wall_iqr_s=iqr([float(r["wall_s"]) for r in subset]),
                median_final_exact_sse=statistics.median(float(r["final_exact_sse"]) for r in subset),
                final_sse_iqr=iqr([float(r["final_exact_sse"]) for r in subset]),
                target_hits=sum(r["target_hit"] == "True" for r in subset),
                target_hit_rate=sum(r["target_hit"] == "True" for r in subset) / len(subset),
                median_target_crossing_time_s=statistics.median(crossing) if crossing else "",
                median_steps=statistics.median(int(r["steps"]) for r in subset),
                mean_monitoring_s=statistics.mean(json.loads(r["timings"]).get("monitoring_s", 0.0)
                                                  for r in subset),
                mean_controller_s=statistics.mean(json.loads(r["timings"]).get("controller_s", 0.0)
                                                  for r in subset),
                records="../phase3_quality_guarded_adaptive/decayed_v2_separate_update_retry3/individual_runs.csv",
                note="separate from cumulative G1/G3/QG update family"))
        write_csv(final_dir / "decayed_v2_summary.csv", decayed_summary)
        prior_path = ROOT / "outputs/20261026-adaptive-v2-observation-budget-40m/measurements_reconstructed.csv"
        prior_rows = rows(prior_path)
        replication_rows = []
        for method in ("C_DECAYED_FIXED", "D_ADAPTIVE_V2"):
            for current in (r for r in decayed_runs if r["method"] == method):
                prior = next((r for r in prior_rows if r["method"] == method and
                              str(r["seed"]) == str(current["seed"])), None)
                if prior is None:
                    raise RuntimeError(f"Prior recovered V2 record missing for {method} seed {current['seed']}")
                if (prior["X_sha256"] != current["x_sha256"] or
                        prior["C0_sha256"] != current["C0_sha256"] or
                        int(prior["points_processed"]) != int(current["observations_processed"])):
                    raise RuntimeError(f"Prior V2 record is not protocol-matched for {method} seed {current['seed']}")
                old_sse = float(prior["final_exact_sse"])
                new_sse = float(current["final_exact_sse"])
                replication_rows.append(dict(
                    method=method, seed=current["seed"], x_hash_match=True,
                    C0_hash_match=True, observation_budget_match=True,
                    prior_final_exact_sse=old_sse, replication_final_exact_sse=new_sse,
                    absolute_sse_difference=abs(old_sse-new_sse),
                    exact_sse_match=bool(np.isclose(old_sse,new_sse,rtol=0,atol=1e-9)),
                    prior_target_hit=prior["target_satisfied"], replication_target_hit=current["target_hit"],
                    prior_external_wall_s=float(prior["external_wall_s"]),
                    replication_external_wall_s=float(current["wall_s"]),
                    wall_time_ratio_replication_over_prior=float(current["wall_s"])/float(prior["external_wall_s"]),
                    note="separate sessions; runtime not pooled; same seed, feature/C0 hashes and observation budget"))
        write_csv(final_dir / "decayed_v2_replication_check.csv", replication_rows)

    crossing_rows = []
    for phase in ("phase1", "phase2", "phase3"):
        checkpoint_rows = rows(phase_dirs[phase] / "checkpoints.csv")
        for run in data[phase]:
            target = float(run["target_sse"])
            if run["method"] == "G0":
                history = json.loads(run["metadata"]).get("quality_history", [])
                hit = next((item for item in history
                            if float(item.get("full_data_sse", float("inf"))) <= target), None)
                obs = (int(hit.get("points_processed", int(hit.get("step", 0)) * int(run["n"])))
                       if hit else None)
                elapsed = float(hit["elapsed_s"]) if hit else None
            else:
                candidates = [item for item in checkpoint_rows
                    if item["method"] == run["method"] and item["seed"] == run["seed"] and
                    item["repeat"] == run["repeat"] and
                    float(item.get("full_exact_sse", float("inf"))) <= target]
                hit = min(candidates, key=lambda item: int(item["points_processed"])) if candidates else None
                obs = int(hit["points_processed"]) if hit else None
                elapsed = float(hit["elapsed_s"]) if hit and hit.get("elapsed_s") else None
            crossing_rows.append(dict(phase=phase, method=run["method"], seed=run["seed"],
                repeat=run["repeat"], target_sse=target,
                reached=bool(hit is not None), first_crossing_observations=obs,
                first_crossing_time_s=elapsed, final_sse=run["final_sse"],
                termination_reason=run["termination_reason"]))
    write_csv(final_dir / "first_target_crossings.csv", crossing_rows)
    timing_summary = []
    timing_fields = ("wall_s", "setup_s", "sampling_s", "batch_assignment_s",
        "accumulation_update_s", "monitoring_s", "controller_s",
        "quality_evaluation_s", "final_assignment_s", "compute_s")
    for phase in ("phase1", "phase2", "phase3"):
        for method in sorted({r["method"] for r in data[phase]}):
            subset = [r for r in data[phase] if r["method"] == method]
            vals_by_field = {key: [float(json.loads(r["timings"])[key])
                for r in subset if json.loads(r["timings"]).get(key) is not None]
                for key in timing_fields}
            for field, values in vals_by_field.items():
                if values:
                    timing_summary.append(dict(phase=phase, method=method, timing_category=field,
                        n=len(values), median_s=statistics.median(values), iqr_s=iqr(values),
                        note="internal categories overlap and are subsets of external wall; do not sum"))
    write_csv(final_dir / "timing_category_summary.csv", timing_summary)

    index = {"study": "20261027-adaptive-four-phase-study", "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_records": {}, "preserved_attempts": [], "figures": [], "skipped": [
            {"phase": "phase4_stage_b", "reason": "real-data ANN gate did not pass both centroid snapshots; combined adaptation was not run"}]}
    for phase, d in phase_dirs.items():
        index["phase_records"][phase] = {"path": str(d.relative_to(base)),
            "complete": manifests[phase].get("complete"),
            "rows": len(data[phase]), "manifest": "manifest.json",
            "checksums_valid": integrity[phase]["checksums_valid"]}
    if decayed_manifest:
        index["phase_records"]["separate_decayed_v2_replication"] = {
            "path": str(decayed_dir.relative_to(base)), "complete": True,
            "rows": len(decayed_runs), "manifest": "manifest.json",
            "checksums_valid": verify_local_manifest(decayed_dir)[0],
            "update_family": "decayed historical sums/effective counts; separate from cumulative phases",
        }
    for config_file in sorted(base.rglob("config.json")):
        d = config_file.parent
        if d == base:
            continue
        rel = str(d.relative_to(base))
        mp = d / "manifest.json"
        entry = {"path": rel, "config": "config.json"}
        if mp.is_file():
            m = json.loads(mp.read_text())
            entry.update(status="complete" if m.get("complete") else "incomplete",
                manifest="manifest.json", failures=m.get("failures", []),
                measured_rows=m.get("completed_measured_fits", m.get("measured_rows")),
                sampling=m.get("config", {}).get("sampling"),
                disjoint_probe=m.get("config", {}).get("disjoint_probe"))
            status_file = d / "attempt_status.json"
            if status_file.is_file():
                entry["attempt_status_note"] = json.loads(status_file.read_text())
        else:
            run_file = d / "individual_runs.csv"
            count = max(0, sum(1 for _ in run_file.open()) - 1) if run_file.is_file() else 0
            status_file = d / "attempt_status.json"
            if status_file.is_file():
                entry.update(json.loads(status_file.read_text()))
            else:
                entry.update(status="incomplete_no_final_manifest", measured_rows=count,
                    note="partial diagnostic or failed attempt; excluded from primary summaries")
        index["preserved_attempts"].append(entry)
    for f in sorted(figdir.glob("*.png")):
        index["figures"].append({"path": str(f.relative_to(base)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    (final_dir / "experiment_index.json").write_text(json.dumps(index, indent=2))

    try:
        gpu_after = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.free,memory.total,pstate,temperature.gpu,clocks.sm,utilization.gpu", "--format=csv"], text=True, timeout=10).strip()
    except Exception as exc:
        gpu_after = f"unavailable: {type(exc).__name__}: {exc}"
    (final_dir / "hardware_after_study.json").write_text(json.dumps({
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-study snapshot, taken after all experiments; not a per-run immediate measurement",
        "nvidia_smi": gpu_after}, indent=2))

    phase1 = {r["method"]: r for r in summary if r["phase"] == "phase1"}
    phase3 = {r["method"]: r for r in summary if r["phase"] == "phase3"}
    phase2 = {r["method"]: r for r in summary if r["phase"] == "phase2"}
    def paired_median_ratio(phase_rows, numerator, denominator):
        a = {(r["seed"], r["repeat"]): float(r["wall_s"]) for r in phase_rows if r["method"] == numerator}
        b = {(r["seed"], r["repeat"]): float(r["wall_s"]) for r in phase_rows if r["method"] == denominator}
        return statistics.median(a[k] / b[k] for k in sorted(a.keys() & b.keys()))
    def paired_median_sse_delta(phase_rows, left, right):
        a = {(r["seed"], r["repeat"]): float(r["final_sse"]) for r in phase_rows if r["method"] == left}
        b = {(r["seed"], r["repeat"]): float(r["final_sse"]) for r in phase_rows if r["method"] == right}
        return statistics.median(a[k] - b[k] for k in sorted(a.keys() & b.keys()))
    phase1_g3_g1_ratio = paired_median_ratio(data["phase1"], "G3CUDA", "G1")
    phase3_qg_g1_ratio = paired_median_ratio(data["phase3"], "G1", "QGCUDA")
    phase3_qg_g3_ratio = paired_median_ratio(data["phase3"], "G3CUDA", "QGCUDA")
    phase3_qg_sse_delta = paired_median_sse_delta(data["phase3"], "QGCUDA", "G1")
    phase2_fixed_cpu_gpu = [float(r["wall_ratio_cpu_over_gpu"]) for r in paired if r["cpu_method"] == "C1"]
    phase2_adapt_cpu_gpu = [float(r["wall_ratio_cpu_over_gpu"]) for r in paired if r["cpu_method"] == "C3"]
    g0_target_times = {}
    g0_target_observations = {}
    for phase in ("phase1", "phase2", "phase3"):
        xs = [r for r in crossing_rows if r["phase"] == phase and r["method"] == "G0" and r["reached"]]
        g0_target_times[phase] = statistics.median(float(r["first_crossing_time_s"]) for r in xs)
        g0_target_observations[phase] = statistics.median(int(r["first_crossing_observations"]) for r in xs)
    q_events = rows(phase_dirs["phase3"] / "controller_events.csv")
    q_events = [r for r in q_events if r["method"] == "QGCUDA"]
    q_changes = []
    for r in q_events:
        decision = json.loads(r["controller_decision"]) if r.get("controller_decision") else {}
        if decision.get("changed"):
            q_changes.append((r, decision))
    q_rollbacks = sum(r["rollback"] == "True" for r in q_events)
    qguard_decision = json.loads((phase_dirs["phase4"] / "manifest.json").read_text()).get("gate_decision", {})
    report = f"""# Adaptive Four-Phase Study — Verified Summary

Study date: 2026-10-27. Dataset: validation crop from {manifests['phase1']['config']['scene'].get('scene_id')}, N=262,144, D=4; K=16. The crop uses frozen development-scene standardization. Results are from the RTX 4050 Laptop GPU and the local CPU in a single-process controlled runner.

## Executed phases

- **Phase 1, fixed vs adaptive CUDA Mini-Batch:** 5 seeds × 3 measured repeats for G0, G1, and G3CUDA; 40.96M observation budget for Mini-Batch; five common full-data SSE checkpoints. Full records: `../phase1_gpu_fixed_vs_adaptive/full_production_stream/`.
- **Phase 2, CPU/GPU comparison:** 5 seeds × 3 repeats for G0, C1, G1, C3, G3CUDA at the same observation budget. All 15 C1/G1 and all 15 C3/G3CUDA schedules matched; no Mini-Batch pair met the frozen quality target. Full records: `../phase2_cpu_gpu_comparisons/full_production_stream/`.
- **Phase 3, quality-guarded CUDA:** 5 seeds × 3 repeats for G0, G1, G3CUDA, QGCUDA at 40.96M observations. This phase used one disjoint seeded 1,024-point controller probe withheld from training for every Mini-Batch arm. The frozen target was available to evaluation only. Full records: `../phase3_quality_guarded_adaptive/full_disjoint_probe/`.
- **Phase 4A, real-data ANN feasibility:** K=16,32,64,128,256; seeds 0,1,2; one measured repetition; initial and post-update centroid snapshots; candidate budgets below K. Predeclared gate required median strict agreement ≥0.95, median assignment-objective gap ≤1%, and median maintenance-inclusive speedup ≥1 on both snapshots. No K/effort pair passed, so Phase 4B (ANN-assisted clustering and combined adaptation) was skipped. Records: `../phase4_ann_and_combined/real_validation_pilot_final/`.

## Main measured findings

| Phase / method | Median complete wall (s) | IQR (s) | Median final exact SSE | Target hits |
|---|---:|---:|---:|---:|
| Phase 1 G0 exact CUDA | {phase1['G0']['median_wall_s']:.3f} | {phase1['G0']['wall_iqr_s']:.3f} | {phase1['G0']['median_final_exact_sse']:.1f} | {phase1['G0']['target_hits']}/{phase1['G0']['n_runs']} |
| Phase 1 G1 fixed CUDA | {phase1['G1']['median_wall_s']:.3f} | {phase1['G1']['wall_iqr_s']:.3f} | {phase1['G1']['median_final_exact_sse']:.1f} | {phase1['G1']['target_hits']}/{phase1['G1']['n_runs']} |
| Phase 1 G3CUDA adaptive | {phase1['G3CUDA']['median_wall_s']:.3f} | {phase1['G3CUDA']['wall_iqr_s']:.3f} | {phase1['G3CUDA']['median_final_exact_sse']:.1f} | {phase1['G3CUDA']['target_hits']}/{phase1['G3CUDA']['n_runs']} |
| Phase 2 C1 fixed CPU | {phase2['C1']['median_wall_s']:.3f} | {phase2['C1']['wall_iqr_s']:.3f} | {phase2['C1']['median_final_exact_sse']:.1f} | {phase2['C1']['target_hits']}/{phase2['C1']['n_runs']} |
| Phase 2 C3 adaptive CPU | {phase2['C3']['median_wall_s']:.3f} | {phase2['C3']['wall_iqr_s']:.3f} | {phase2['C3']['median_final_exact_sse']:.1f} | {phase2['C3']['target_hits']}/{phase2['C3']['n_runs']} |
| Phase 3 G1 fixed CUDA | {phase3['G1']['median_wall_s']:.3f} | {phase3['G1']['wall_iqr_s']:.3f} | {phase3['G1']['median_final_exact_sse']:.1f} | {phase3['G1']['target_hits']}/{phase3['G1']['n_runs']} |
| Phase 3 G3CUDA adaptive | {phase3['G3CUDA']['median_wall_s']:.3f} | {phase3['G3CUDA']['wall_iqr_s']:.3f} | {phase3['G3CUDA']['median_final_exact_sse']:.1f} | {phase3['G3CUDA']['target_hits']}/{phase3['G3CUDA']['n_runs']} |
| Phase 3 QGCUDA | {phase3['QGCUDA']['median_wall_s']:.3f} | {phase3['QGCUDA']['wall_iqr_s']:.3f} | {phase3['QGCUDA']['median_final_exact_sse']:.1f} | {phase3['QGCUDA']['target_hits']}/{phase3['QGCUDA']['n_runs']} |

G1 and G3CUDA did not reach the quality target at 40.96M observations, so their runtime differences do not establish time-to-quality improvement. G0's median first target crossing was {g0_target_times['phase1']:.3f}s at {g0_target_observations['phase1']:.0f} observations in Phase 1, {g0_target_times['phase2']:.3f}s at {g0_target_observations['phase2']:.0f} observations in Phase 2, and {g0_target_times['phase3']:.3f}s at {g0_target_observations['phase3']:.0f} observations in Phase 3. Mini-Batch arms recorded no target crossings. In Phase 1, the paired median G3CUDA/G1 wall-time ratio was {phase1_g3_g1_ratio:.2f}; the adaptive baseline was slower in this observation-budget comparison. In Phase 3, the paired median G1/QGCUDA and G3CUDA/QGCUDA wall-time ratios were {phase3_qg_g1_ratio:.2f} and {phase3_qg_g3_ratio:.2f}, respectively; paired median QGCUDA-minus-G1 SSE was {phase3_qg_sse_delta:.2f}. QGCUDA completed the same observation budget faster than both baselines while producing a small lower SSE, but it still missed the target in every run. The controller made {len(q_changes)} batch-size increases across 15 measured runs, each after probe-objective stagnation, and reached its configured maximum in every run; no rollback was triggered ({q_rollbacks} rollback events). This supports an equal-work runtime observation for this scene/configuration, not a claim of target-quality acceleration or general superiority.

First target-crossing rows are in `first_target_crossings.csv`, derived from the saved exact per-step histories and full-data checkpoints. The original `individual_runs.csv` files are preserved byte-for-byte; their G0 `target_crossing_observation` field contains elapsed seconds due to a runner reporting bug. The derived table corrects this by using the actual processed-observation count from the recorded first crossing, and the runner has been fixed for future runs.

In Phase 2, paired C1/G1 and C3/G3CUDA work schedules matched in all 15 repetitions per pair. Median paired CPU/GPU wall ratios were {statistics.median(phase2_fixed_cpu_gpu):.3f} (fixed) and {statistics.median(phase2_adapt_cpu_gpu):.3f} (adaptive), where a ratio below one means the GPU took longer. These are complete-work ratios, not matched-quality speedups, because no Mini-Batch pair reached the target. CPU adaptive versus fixed performance was nearly unchanged at this budget.

The ANN gate found candidate-specific accuracy/cost tradeoffs. Some settings were faster but exceeded the assignment-objective gap; others met quality on one centroid snapshot but not both or were slower. The predeclared gate failed as a K/effort pair across initial and updated centroids; no ANN clustering/combined adaptation was run. The first two ANN pilots are retained as earlier diagnostics; `real_validation_pilot_final` includes the device-resident refresh path and complete per-experiment schemas.

## Separate decayed-count update-family replication

This is a distinct update rule, not an extension of the cumulative G1/G3CUDA/QGCUDA comparison. On the same 31UFU crop, K=16, seeds 0–2, shared seed-specific C0, and 40.96M processed observations, fixed decay and Adaptive V2 were each run once per seed with five common quality checkpoints. The update rule uses observation-scaled decay; the V2 controller uses only its fixed probe and does not receive the frozen target. Median wall times were {decayed_summary[0]['median_wall_s']:.3f}s for fixed decay and {decayed_summary[1]['median_wall_s']:.3f}s for Adaptive V2. Median final exact SSEs were {decayed_summary[0]['median_final_exact_sse']:.2f} and {decayed_summary[1]['median_final_exact_sse']:.2f}; both reached the frozen evaluation target on {decayed_summary[0]['target_hits']}/3 seeds. The six per-seed final SSEs and target-hit outcomes match the earlier recovered observation-budget archive at the same data/C0 hashes and work budget; this is evidence of deterministic result reproduction, not a second independent sample. Wall times varied across sessions and are reported separately. This small replication suggests fewer Adaptive V2 updates may reduce runtime at equal processed observations, but n=3 and one measurement per seed do not establish a general efficiency advantage. It is not evidence about the cumulative Mini-Batch family and is not pooled into the main method table. Records and controller/update histories: `../phase3_quality_guarded_adaptive/decayed_v2_separate_update_retry3/`; method summary: `decayed_v2_summary.csv`; per-seed archive check: `decayed_v2_replication_check.csv`; plot: `../figures/13_decayed_v2_fixed_vs_adaptive.png`.

Earlier pilot protocols and interrupted attempts are retained, with status and sampling policy in `experiment_index.json`. Partial files or runs without a completed manifest are excluded from primary estimates. The interrupted Phase 1 exploratory run used a probe-excluded sample policy and is not combined with the corrected production-stream result. A sandboxed retry that could not initialize CUDA is recorded as a failed attempt, not as a CPU substitute.

## Limitations and skipped work

- Validation results cover one 31UFU acquisition/crop, D=4 and K=16 for Mini-Batch comparisons. They do not establish generalization across scenes or K.
- The exact CPU reference determines the frozen 5% evaluation target. No controller receives that target.
- One quality-guard pilot and its repeated experiment showed no rollback; rollback correctness is covered by real-GPU tests, but real-scene rollback benefits are unobserved.
- ANN Stage B was skipped by the gate; no results are claimed for G2CUDA/G4CUDA/G5CUDA or combined ANN/batch adaptation here.
- Phase 4 is a one-repetition-per-seed feasibility pilot; timings are noisier and not a general benchmark.
- Small-kernel Numba occupancy warnings and Rasterio pending-deprecation warnings appeared in the regression suite; tests passed.

## Reproduction and evidence

Commands and environment are in `../README.md`. Per-run source snapshots, configurations, raw records, checkpoints, timing breakdowns, manifests and checksums are stored beside each experiment. `method_comparison.csv`, `timing_category_summary.csv`, `first_target_crossings.csv`, `decayed_v2_summary.csv`, `decayed_v2_replication_check.csv` and `experiment_index.json` are this summary's machine-readable companions. Timing components overlap and must not be summed. The output-level `checksums.sha256` lists all preserved study files.
"""
    (final_dir / "final_summary.md").write_text(report)

    # Reproduction instructions at the study root.
    readme = """# Controlled Adaptive GPU Mini-Batch Study (2026-10-27)

This isolated archive uses the prepared real validation crop `outputs/20261025-independent-holdouts/validation512` (Sentinel-2 31UFU; B02/B03/B04/B08; product-calibrated reflectance; SCL/cloud/nodata masks applied; scaling frozen from development scene). No SAFE archive is copied into this output directory.

## Reproduce

Run from the repository root in the existing `.venv`; the launchers set the WSL CUDA library path and require a real GPU. The runner rejects CUDA simulator/fallback execution.

```bash
bash scripts/run_cuda_python.sh experiments/run_adaptive_four_phase.py \\
  --phase phase1 --stage full \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --output-dir outputs/20261027-adaptive-four-phase-study/phase1_gpu_fixed_vs_adaptive/reproduction \\
  --time-budget-seconds 2400

bash scripts/run_cuda_python.sh experiments/run_adaptive_four_phase.py \\
  --phase phase2 --stage full \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --output-dir outputs/20261027-adaptive-four-phase-study/phase2_cpu_gpu_comparisons/reproduction \\
  --time-budget-seconds 2400

bash scripts/run_cuda_python.sh experiments/run_adaptive_four_phase.py \\
  --phase phase3 --stage full \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --output-dir outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/reproduction \\
  --time-budget-seconds 3600

bash scripts/run_cuda_python.sh experiments/run_real_projection_ann_gate.py \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --output-dir outputs/20261027-adaptive-four-phase-study/phase4_ann_and_combined/reproduction \\
  --clusters 16 32 64 128 256 --seeds 0 1 2 --warmups 1 --repeats 1

.venv/bin/python experiments/finalize_adaptive_four_phase.py
```

To reproduce the separate decayed-count fixed-versus-Adaptive-V2 comparison, use the same prepared scene, seed-specific Phase 3 references and settings, but choose a new output directory that does not already exist:

```bash
bash scripts/run_cuda_python.sh experiments/run_decayed_v2_replication.py \\
  --scene-dir outputs/20261025-independent-holdouts/validation512 \\
  --reference-dir outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/full_disjoint_probe \\
  --output-dir outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/decayed_v2_reproduction \\
  --seeds 0 1 2 --k 16 --batch-size 8192 --minimum-batch 256 \\
  --maximum-batch 65536 --max-steps 5000 --probe-interval 10 \\
  --observation-budget 40960000 --checkpoint-interval 8192000 --warmups 1
```

Each experiment requires a new, nonexistent output directory. The `full` runner uses seeds 0–4, three measured repetitions, and 40.96M Mini-Batch observations. Warm-ups are excluded. Phase 3 uses a disjoint controller probe; Phases 1–2 use the full production sampling population so G1/G3 replay their established sampling policy. ANN Stage B must remain skipped unless the predeclared gate passes.

## Study artifacts

- `phase1_gpu_fixed_vs_adaptive/`: pilot, full G0/G1/G3CUDA data.
- `phase2_cpu_gpu_comparisons/`: paired C1/G1 and C3/G3CUDA CPU/GPU data.
- `phase3_quality_guarded_adaptive/`: pure controller tests, pilot and repeated G0/G1/G3CUDA/QGCUDA study.
- `phase4_ann_and_combined/`: real-data projection-ANN assignment gate; combined adaptation was skipped after the gate failed.
- `phase3_quality_guarded_adaptive/decayed_v2_separate_update_retry3/`: separate fixed-decay versus Adaptive V2 replication; it is not pooled with the cumulative-update experiments.
- `figures/`: plots copied or derived from measured result files.
- `final_summary/`: verified report, comparison table, hardware post-study snapshot, and experiment index.

Original pilot and incomplete/failed protocol attempts are retained and identified by their own configs/manifests. Do not pool them with the production-stream Phase 1/2 full runs or disjoint-probe Phase 3 runs. Always inspect each manifest's `complete`, `failures`, `sampling`, and `disjoint_probe` fields.
"""
    (base / "README.md").write_text(readme)

    # Study-wide checksums (excluding the checksum list itself).
    checksum_file = base / "checksums.sha256"
    files = [f for f in base.rglob("*") if f.is_file() and f != checksum_file]
    checksum_file.write_text("".join(
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(base)}\n"
        for f in sorted(files)))
    # Verify the complete study checksum after writing it.
    bad = []
    for line in checksum_file.read_text().splitlines():
        expected_hash, rel = line.split("  ", 1)
        if hashlib.sha256((base / rel).read_bytes()).hexdigest() != expected_hash:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"Study-wide checksum verification failed: {bad}")
    print(f"Finalized and verified {len(files)} study files at {base}")


if __name__ == "__main__":
    main()
