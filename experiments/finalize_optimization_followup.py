"""Assemble and audit the already completed PCAP optimization studies.

This finalizer does not execute clustering fits. It validates the source
archives, copies their recorded evidence byte-for-byte into a dated bundle,
and produces derived summaries/figures from the saved rows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/20261027-adaptive-optimization-followup"
PHASE_A = ROOT / "outputs/20261027-qg-batchsize-ablation"
FOUR_PHASE = ROOT / "outputs/20261027-adaptive-four-phase-study"
DECAY = ROOT / "outputs/20261027-decay-memory-study"
OLD_DECAY_DIAG = ROOT / "outputs/20261026-validation-decayed-count-k16-b8192-seed0"
PROFILE = Path.home() / "pcap-profiling-archive/cuda-baseline-N32768-20261020"

PHASE_A_METHODS = {
    "FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536",
    "QGCUDA", "G3CUDA", "G0",
}
PHASE_C_METHODS = {
    "FIXED_DECAY_B8192", "FIXED_DECAY_B16384", "FIXED_DECAY_B32768",
    "FIXED_DECAY_B65536", "ADAPTIVE_V2", "G0_EXACT_CUDA",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_checksum_list(directory: Path, checksum_file: str = "checksums.sha256") -> int:
    entries = (directory / checksum_file).read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in entries:
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.lstrip("* ")
        target = directory / rel
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Checksum failure: {target}")
        checked += 1
    if checked == 0:
        raise ValueError(f"Empty checksum list: {directory / checksum_file}")
    return checked


def validate_phase_a(directory: Path = PHASE_A) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    rows = read_csv(directory / "individual_runs.csv")
    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError("Phase A source archive is incomplete or has fit failures")
    if len(rows) != 105 or {r["method"] for r in rows} != PHASE_A_METHODS:
        raise ValueError("Phase A is not the expected 105-fit, seven-method matrix")
    counts = {(m, s): 0 for m in PHASE_A_METHODS for s in range(5)}
    c0_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        pair = (row["method"], row["seed"])
        counts[pair] += 1
        c0_by_pair.setdefault((row["seed"], row["repeat"]), set()).add(row["C0_sha256"])
        if row["X_sha256"] != config["x_sha256"]:
            raise ValueError("Phase A input hash differs within the archived matrix")
        if row["method"] != "G0":
            if row["processed_observations"] != str(config["observation_budget"]):
                raise ValueError("Phase A Mini-Batch budget mismatch")
            if row["termination_reason"] != "observation_budget":
                raise ValueError("Phase A Mini-Batch arm did not stop at its budget")
            if row["target_is_evaluation_only"] != "True":
                raise ValueError("Phase A target was not marked evaluator-only")
    if any(n != 3 for n in counts.values()):
        raise ValueError("Phase A does not have three repetitions for every method/seed")
    if any(len(v) != 1 for v in c0_by_pair.values()):
        raise ValueError("Phase A methods do not share C0 for every seed/repetition")
    if not config.get("disjoint_probe") or config.get("probe_size") != 1024:
        raise ValueError("Phase A lacks the common disjoint 1024-point probe")
    if config.get("observation_budget") != 40_960_000:
        raise ValueError("Unexpected Phase A observation budget")
    return {"runs": len(rows), "methods": sorted(PHASE_A_METHODS), "seeds": 5,
            "repetitions_per_seed_method": 3, "manifest": manifest, "config": config,
            "rows": rows}


def validate_phase_b(directory: Path = DECAY / "phase0_convergence_diagnostic_retry1") -> dict:
    runs = read_csv(directory / "individual_runs.csv")
    checkpoints = read_csv(directory / "checkpoints.csv")
    trace = read_csv(directory / "update_trace.csv")
    if len(runs) != 6 or len(checkpoints) != 30:
        raise ValueError("Phase B must contain three matched seeds, two policies, five checkpoints")
    if any(r["sampling_stream_matches"] != "True" for r in runs):
        raise ValueError("Matched cumulative/decay sampling stream was not confirmed")
    trace_hashes = {}
    for r in trace:
        trace_hashes.setdefault((r["seed"], r["method"]), []).append(r["sample_indices_sha256"])
    for seed in ("0", "1", "2"):
        if trace_hashes[(seed, "CUMULATIVE")] != trace_hashes[(seed, "DECAYED_FIXED_8192")]:
            raise ValueError(f"Phase B sampled index streams differ for seed {seed}")
    return {"runs": runs, "checkpoints": checkpoints, "trace": trace}


def validate_phase_c(directory: Path = DECAY / "phase2_validation") -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    rows = read_csv(directory / "individual_runs.csv")
    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError("Phase C source archive is incomplete or has fit failures")
    if len(rows) != 90 or {r["method"] for r in rows} != PHASE_C_METHODS:
        raise ValueError("Phase C is not the expected 90-fit six-method matrix")
    counts = {(m, s): 0 for m in PHASE_C_METHODS for s in range(5)}
    c0_by_pair: dict[tuple[str, str], set[str]] = {}
    x_hashes: set[str] = set()
    for row in rows:
        counts[(row["method"], int(row["seed"]))] += 1
        x_hashes.add(row["X_sha256"])
        c0_by_pair.setdefault((row["seed"], row["repeat"]), set()).add(row["C0_sha256"])
        if row["method"] != "G0_EXACT_CUDA":
            if int(row["observations_processed"]) != int(config["observation_budget"]):
                raise ValueError("Phase C Mini-Batch budget mismatch")
            if row["termination_reason"] != "observation_budget":
                raise ValueError("Phase C Mini-Batch arm did not stop at its budget")
    if any(n != 3 for n in counts.values()):
        raise ValueError("Phase C does not have three repetitions for every method/seed")
    if len(x_hashes) != 1 or any(len(v) != 1 for v in c0_by_pair.values()):
        raise ValueError("Phase C methods do not share X and C0 within each seed/repetition")
    if config.get("observation_budget") != 40_960_000 or config.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("Phase C protocol differs from its frozen budget/seeds")
    if any(r.get("target_evaluation_only") != "True" for r in rows):
        raise ValueError("Phase C target was not marked evaluator-only")
    return {"runs": len(rows), "methods": sorted(PHASE_C_METHODS), "seeds": 5,
            "repetitions_per_seed_method": 3, "manifest": manifest, "config": config,
            "rows": rows}


def median_iqr(values: list[float]) -> tuple[float, float]:
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))


def summarize_timing(rows: list[dict[str, str]], methods: set[str], wall_key: str,
                     field_map: dict[str, str], method_key: str = "method") -> list[dict]:
    result = []
    for method in sorted(methods):
        subset = [r for r in rows if r[method_key] == method]
        if not subset:
            continue
        item = {"method": method, "n_runs": len(subset)}
        item["external_wall_median_s"], item["external_wall_iqr_s"] = median_iqr(
            [float(r[wall_key]) for r in subset])
        for output, source in field_map.items():
            vals = [float(r[source]) for r in subset if r.get(source) not in (None, "")]
            if vals:
                item[output + "_median_s"], item[output + "_iqr_s"] = median_iqr(vals)
            else:
                item[output + "_median_s"] = ""
                item[output + "_iqr_s"] = ""
        result.append(item)
    return result


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_figures(phase_a_rows: list[dict[str, str]], phase_a_checkpoints: list[dict[str, str]],
                 phase_c_rows: list[dict[str, str]], phase_c_checkpoints: list[dict[str, str]],
                 timing_rows: list[dict], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    generated = []
    fixed = ["FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536"]
    all_a = sorted({r["method"] for r in phase_a_rows})
    qg = [r for r in phase_a_rows if r["method"] == "QGCUDA"]
    # 1: complete runtime vs batch width; adaptive methods are reference points.
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    x = np.array([8192, 16384, 32768, 65536], dtype=float)
    for method in fixed:
        vals = [float(r["external_wall_s"]) for r in phase_a_rows if r["method"] == method]
        ax.scatter([int(method.split("_")[1])] * len(vals), vals, alpha=.55, s=24)
    med = [np.median([float(r["external_wall_s"]) for r in phase_a_rows if r["method"] == m]) for m in fixed]
    ax.plot(x, med, marker="o", label="Fixed median")
    ax.axhline(float(np.median([float(r["external_wall_s"]) for r in qg])), color="tab:orange", ls="--", label="QGCUDA median")
    ax.set_xscale("log", base=2); ax.set_xticks(x, ["8,192", "16,384", "32,768", "65,536"])
    ax.set(xlabel="Fixed batch size", ylabel="Complete fit wall time (s)", title="Cumulative Mini-Batch runtime on 31UFU")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "01_runtime_vs_fixed_batch.png", dpi=200); plt.close(fig)
    generated.append("01_runtime_vs_fixed_batch.png")

    # 2: final SSE versus batch size, with all individual runs.
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for method in fixed:
        batch = int(method.split("_")[1])
        vals = [float(r["final_exact_sse"]) for r in phase_a_rows if r["method"] == method]
        ax.scatter([batch] * len(vals), vals, alpha=.52, s=24)
        ax.plot([batch], [np.median(vals)], marker="D", color="black", ms=5)
    qvals = [float(r["final_exact_sse"]) for r in qg]
    ax.axhline(np.median(qvals), color="tab:orange", ls="--", label="QGCUDA median")
    ax.set_xscale("log", base=2); ax.set_xticks(x, ["8,192", "16,384", "32,768", "65,536"])
    ax.set(xlabel="Fixed batch size", ylabel="Final exact full-data SSE", title="Quality versus batch size; all MB arms missed frozen target")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "02_sse_vs_fixed_batch.png", dpi=200); plt.close(fig)
    generated.append("02_sse_vs_fixed_batch.png")

    # 3: QG vs best fixed comparison, individual paired runs.
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    b = [r for r in phase_a_rows if r["method"] == "FIXED_65536"]
    q_by = {(r["seed"], r["repeat"]): r for r in qg}
    b_by = {(r["seed"], r["repeat"]): r for r in b}
    for key in sorted(q_by):
        ax.plot([float(b_by[key]["external_wall_s"]), float(q_by[key]["external_wall_s"])],
                [float(b_by[key]["final_exact_sse"]), float(q_by[key]["final_exact_sse"])],
                color="0.7", lw=.8, zorder=1)
    ax.scatter([float(r["external_wall_s"]) for r in b], [float(r["final_exact_sse"]) for r in b], label="Fixed B=65,536", s=35)
    ax.scatter([float(r["external_wall_s"]) for r in qg], [float(r["final_exact_sse"]) for r in qg], label="QGCUDA", s=35)
    ax.set(xlabel="Complete fit wall time (s; lower is better)", ylabel="Final exact full-data SSE (lower is better)", title="Paired QGCUDA versus fastest fixed batch")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "03_qg_vs_best_fixed.png", dpi=200); plt.close(fig)
    generated.append("03_qg_vs_best_fixed.png")

    # 4: common SSE checkpoints, all medians by recorded checkpoint number.
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for method in [*fixed, "QGCUDA", "G3CUDA"]:
        sub = [r for r in phase_a_checkpoints if r["method"] == method]
        if not sub: continue
        grouped = {}
        for row in sub: grouped.setdefault(int(row["checkpoint_index"]), []).append(float(row["full_exact_sse"]))
        xx = sorted(grouped); yy = [np.median(grouped[i]) for i in xx]
        obs = [np.median([int(r["processed_observations"]) for r in sub if int(r["checkpoint_index"]) == i]) / 1e6 for i in xx]
        ax.plot(obs, yy, marker="o", ms=3, label=method)
    ax.set(xlabel="Processed observations (millions)", ylabel="Exact full-data SSE", title="Common evaluation checkpoints (31UFU)")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(out / "04_cumulative_sse_vs_observations.png", dpi=200); plt.close(fig)
    generated.append("04_cumulative_sse_vs_observations.png")

    # 5: QG controller batch schedule, one illustrative measured run only.
    example = next((r for r in qg if r["seed"] == "0" and r["repeat"] == "0"), None)
    if example:
        sched = json.loads(example["batch_schedule"])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.step([float(r["points_processed"]) / 1e6 for r in sched], [int(r["actual_batch_size"]) for r in sched], where="post")
        ax.set(xlabel="Processed observations (millions)", ylabel="Actual batch size", title="QGCUDA schedule (seed 0, repetition 0)")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "05_qg_batch_trajectory.png", dpi=200); plt.close(fig)
        generated.append("05_qg_batch_trajectory.png")

    # 6: fixed-decay vs V2 on validation, all individual repetitions.
    c_fixed = [r for r in phase_c_rows if r["method"] == "FIXED_DECAY_B65536"]
    c_v2 = [r for r in phase_c_rows if r["method"] == "ADAPTIVE_V2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax, field, label in ((axes[0], "external_wall_s", "Complete fit wall (s)"),
                             (axes[1], "final_exact_sse", "Final exact SSE")):
        ax.scatter(np.zeros(len(c_fixed)), [float(r[field]) for r in c_fixed], label="Fixed decay B=65,536", alpha=.7)
        ax.scatter(np.ones(len(c_v2)), [float(r[field]) for r in c_v2], label="Adaptive V2", alpha=.7)
        ax.set_xticks([0, 1], ["Fixed decay", "Adaptive V2"]); ax.set_ylabel(label); ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8); fig.suptitle("31UFU; same 40.96M budget; decayed update family")
    fig.tight_layout(); fig.savefig(out / "06_fixed_decay_vs_v2.png", dpi=200); plt.close(fig)
    generated.append("06_fixed_decay_vs_v2.png")

    # 7: measured internal timing categories. These are not stacked or added.
    timing_methods = ["FIXED_8192", "FIXED_65536", "QGCUDA", "G3CUDA"]
    categories = [("sampling_s", "Sampling"), ("batch_assignment_s", "Batch assignment"),
                  ("accumulation_update_s", "Accumulation/update"), ("monitoring_s", "Monitoring")]
    lookup = {r["method"]: r for r in timing_rows}
    fig, ax = plt.subplots(figsize=(9.0, 5.0)); width = .18; pos = np.arange(len(timing_methods))
    for j, (key, label) in enumerate(categories):
        vals = [float(lookup[m].get(key + "_median_s", 0)) for m in timing_methods]
        ax.bar(pos + (j - 1.5) * width, vals, width, label=label)
    ax.set_xticks(pos, timing_methods); ax.set_ylabel("Median internal wall category (s)")
    ax.set_title("Instrumented operation groups; not per-kernel and not additive to total wall")
    ax.grid(axis="y", alpha=.25); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(out / "07_grouped_gpu_timing.png", dpi=200); plt.close(fig)
    generated.append("07_grouped_gpu_timing.png")

    # 8: frozen-target hit rates, show zero-hit methods rather than omit misses.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, rows, methods, title in (
        (axes[0], phase_a_rows, all_a, "Cumulative family"),
        (axes[1], phase_c_rows, sorted(PHASE_C_METHODS), "Decayed family"),
    ):
        rates = [sum(r["target_hit"] == "True" for r in rows if r["method"] == m) /
                 max(1, sum(r["method"] == m for r in rows)) for m in methods]
        ax.bar(np.arange(len(methods)), rates)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=55, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Frozen target hit rate"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Target is evaluator-only; misses remain visible")
    fig.tight_layout(); fig.savefig(out / "08_frozen_target_hit_rates.png", dpi=200); plt.close(fig)
    generated.append("08_frozen_target_hit_rates.png")

    # 9: time-to-target; only measured crossings plotted, misses explicitly annotated.
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    methods = ["FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536", "QGCUDA", "G3CUDA", "G0"]
    for i, m in enumerate(methods):
        subset = [r for r in phase_a_rows if r["method"] == m]
        hits = [float(r["first_target_crossing_time_s"]) for r in subset if r.get("first_target_crossing_time_s") not in ("", "None")]
        if hits: ax.scatter([i] * len(hits), hits, s=18, label=m)
        else: ax.text(i, .05, "miss", rotation=90, ha="center", va="bottom", color="tab:red", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right"); ax.set_ylabel("First observed target-crossing time (s)")
    ax.set_title("Cumulative-family target crossings; no crossings are not imputed")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "09_time_to_quality.png", dpi=200); plt.close(fig)
    generated.append("09_time_to_quality.png")
    return generated


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_checksums(root: Path) -> int:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    lines = [f"{sha256(p)}  {p.relative_to(root).as_posix()}" for p in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def build_archive() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUT}")
    a = validate_phase_a()
    b = validate_phase_b()
    c = validate_phase_c()
    for root in (PHASE_A, DECAY):
        verify_checksum_list(root)
    OUT.mkdir(parents=True)
    phase_a_out = OUT / "phaseA_fixed_batch_sweep"
    phase_b_out = OUT / "phaseB_quality_diagnostics"
    phase_c_out = OUT / "phaseC_adaptive_decay"
    phase_d_out = OUT / "phaseD_gpu_performance"
    for d in (phase_a_out, phase_b_out, phase_c_out, phase_d_out, OUT / "figures", OUT / "final_summary"):
        d.mkdir(parents=True, exist_ok=True)

    # Preserve source files byte-for-byte, while avoiding any SAFE/raw imagery copy.
    copy_tree(PHASE_A, phase_a_out / "source_archive")
    copy_tree(DECAY / "phase0_convergence_diagnostic_retry1", phase_b_out / "matched_cumulative_vs_decay_trace")
    if OLD_DECAY_DIAG.is_dir():
        copy_tree(OLD_DECAY_DIAG, phase_b_out / "historical_5000_update_single_seed_diagnostic")
    copy_tree(DECAY / "phase1_development_retry1", phase_c_out / "development_pilot_31UFS")
    copy_tree(DECAY / "phase2_validation", phase_c_out / "validation_31UFU")
    _copy_file(DECAY / "frozen_validation_protocol.json", phase_c_out / "frozen_validation_protocol.json")
    for name in ("summary.md", "method_summary.csv", "paired_v2_vs_fixed.csv", "timing_summary.csv",
                 "controller_summary.json", "cumulative_vs_decay_diagnostic_summary.csv",
                 "source_and_measurement_corrections.json", "individual_runs_reconciled.csv"):
        source = DECAY / "final_summary" / name
        if source.is_file():
            _copy_file(source, phase_c_out / "derived" / name)

    # Include prior short QG pilot and full phase-3 comparison as supplemental provenance.
    pilot_src = FOUR_PHASE / "phase3_quality_guarded_adaptive" / "pilot_40m_disjoint_probe"
    if pilot_src.is_dir():
        copy_tree(pilot_src, phase_a_out / "prior_3_seed_quality_guard_pilot")
    phase3_src = FOUR_PHASE / "phase3_quality_guarded_adaptive" / "full_disjoint_probe"
    if phase3_src.is_dir():
        _copy_file(phase3_src / "manifest.json", phase_a_out / "historical_full_phase3_manifest.json")
        _copy_file(phase3_src / "method_summary.csv", phase_a_out / "historical_full_phase3_method_summary.csv")

    # Phase D: copy only existing timing evidence and the prior profiler blocker record.
    _copy_file(PHASE_A / "timing_breakdown.csv", phase_d_out / "cumulative_timing_breakdown.csv")
    _copy_file(DECAY / "phase2_validation" / "timing_breakdown.csv", phase_d_out / "decayed_timing_breakdown.csv")
    if (PROFILE / "investigation.json").is_file():
        copy_tree(PROFILE, phase_d_out / "prior_nsight_attempt")

    phase_a_checkpoints = read_csv(PHASE_A / "checkpoints.csv")
    phase_c_checkpoints = read_csv(DECAY / "phase2_validation" / "checkpoints.csv")
    phase_a_timing = summarize_timing(read_csv(PHASE_A / "timing_breakdown.csv"), PHASE_A_METHODS,
        "external_wall_s", {"sampling_s":"sampling_s", "batch_assignment_s":"batch_assignment_s",
        "accumulation_update_s":"accumulation_update_s", "monitoring_s":"monitoring_s",
        "controller_s":"controller_s", "quality_evaluation_s":"quality_evaluation_s",
        "final_assignment_s":"final_assignment_s"})
    phase_c_timing = summarize_timing(read_csv(DECAY / "phase2_validation" / "timing_breakdown.csv"), PHASE_C_METHODS,
        "external_wall_s", {"sampling_s":"internal_sampling_s", "sample_provenance_s":"internal_sampling_provenance_s",
        "batch_assignment_s":"internal_batch_assignment_s", "accumulation_update_s":"internal_accumulation_update_s",
        "monitoring_s":"internal_monitoring_s", "controller_s":"internal_controller_s",
        "quality_evaluation_s":"internal_quality_evaluation_s", "final_assignment_s":"internal_final_assignment_s"})
    write_csv(OUT / "final_summary" / "phaseA_timing_summary.csv", phase_a_timing)
    write_csv(OUT / "final_summary" / "phaseC_timing_summary.csv", phase_c_timing)
    phase_a_runs = a["rows"]
    phase_c_runs = [dict(r, external_wall_s=r["wall_s"]) for r in c["rows"]]
    figures = make_figures(phase_a_runs, phase_a_checkpoints, phase_c_runs, phase_c_checkpoints,
                           phase_a_timing, OUT / "figures")

    # Compact independent derived tables; original CSVs remain byte-identical copies.
    write_csv(OUT / "final_summary" / "phaseA_method_summary.csv", read_csv(PHASE_A / "method_summary.csv"))
    write_csv(OUT / "final_summary" / "phaseA_paired_comparisons.csv", read_csv(PHASE_A / "paired_comparisons.csv"))
    write_csv(OUT / "final_summary" / "phaseC_method_summary.csv", read_csv(DECAY / "final_summary" / "method_summary.csv"))
    write_csv(OUT / "final_summary" / "phaseC_paired_v2_vs_fixed.csv", read_csv(DECAY / "final_summary" / "paired_v2_vs_fixed.csv"))
    write_csv(OUT / "final_summary" / "phaseB_matched_update_diagnostic.csv", read_csv(DECAY / "final_summary" / "cumulative_vs_decay_diagnostic_summary.csv"))

    profile_note = {
        "status": "no usable per-kernel timeline available",
        "source": str(PROFILE / "investigation.json"),
        "interpretation": "Existing host-instrumented grouped timing fields are reported. They include assignment, accumulation/update, sampling and monitoring scopes, but do not isolate individual kernels, launch gaps, synchronization, or H2D/D2H transfers. No optimization was attempted because the available Nsight Systems capture failed to yield a trace and grouped intervals are not kernel timings.",
        "optimization_changes": "none; original implementations unchanged",
    }
    (phase_d_out / "profiling_and_optimization_status.json").write_text(json.dumps(profile_note, indent=2) + "\n")

    report = _render_report(a, b, c, phase_a_timing, phase_c_timing, figures)
    (OUT / "final_summary" / "final_findings.md").write_text(report, encoding="utf-8")
    (OUT / "README.md").write_text(_render_readme(figures), encoding="utf-8")
    index = {
        "study": "PCAP adaptive optimization follow-up",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "experiment-adaptive-v2",
        "contains_new_fits": False,
        "reuse_reason": "The requested controlled cumulative 105-fit and fixed-decay 90-fit studies had already been completed under the requested protocols; no duplicate runs were made.",
        "stages": {
            "phaseA_fixed_batch_sweep": {"status":"validated existing 105-fit real-GPU study", "source":str(PHASE_A.relative_to(ROOT)), "runs":a["runs"]},
            "phaseB_quality_diagnostics": {"status":"validated existing matched 3-seed diagnostic", "source":str((DECAY / 'phase0_convergence_diagnostic_retry1').relative_to(ROOT)), "runs":len(b["runs"])},
            "phaseC_adaptive_decay": {"status":"validated existing dev pilot and 90-fit real-GPU validation", "source":str((DECAY / 'phase2_validation').relative_to(ROOT)), "runs":c["runs"], "quality_aware_rollback_variant":"skipped; no probe deterioration events were recorded to motivate rollback"},
            "phaseD_gpu_performance": {"status":"grouped timing reanalysis; per-kernel profiling unavailable", "source":str(PHASE_A.relative_to(ROOT))},
        },
        "figures": figures,
        "source_archive_manifests": {
            "phaseA":sha256(PHASE_A / "manifest.json"),
            "phaseC":sha256(DECAY / "phase2_validation" / "manifest.json"),
            "frozen_protocol":sha256(DECAY / "frozen_validation_protocol.json"),
        },
        "limitations": [
            "Phase A archived seeded sampling policies and common training eligibility, but not per-batch sample-index hashes; pointwise sample equality across batch widths cannot be asserted.",
            "No new Phase A pilot was run because the complete controlled 105-fit matrix was already available; its prior pilot was not identical to the full fixed-width sweep.",
            "No schedule replay was run because Adaptive V2 did not pass its predeclared advantage gate over the fastest fixed-decay arm.",
            "No per-kernel timeline or isolated transfer/launch timings are available; no low-level GPU optimization was attempted.",
            "All evidence is from the declared crops, D=4, K=16 and five seeds where applicable; repetitions reuse deterministic sampling streams.",
        ],
    }
    (OUT / "experiment_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    count = _write_checksums(OUT)
    checked = verify_checksum_list(OUT)
    if count != checked:
        raise RuntimeError(f"Archive checksum verification count mismatch: {count} vs {checked}")
    return OUT


def _fmt(rows: list[dict], method: str, key: str) -> str:
    r = next(x for x in rows if x["method"] == method)
    return f"{float(r[key]):,.3f}"


def _render_report(a: dict, b: dict, c: dict, phase_a_timing: list[dict], phase_c_timing: list[dict], figures: list[str]) -> str:
    a_rows, c_rows = a["rows"], c["rows"]
    a_methods = read_csv(PHASE_A / "method_summary.csv")
    c_methods = read_csv(DECAY / "final_summary" / "method_summary.csv")
    # Use verified archived summary values but derive comparisons again from paired records.
    paired = read_csv(PHASE_A / "paired_comparisons.csv")
    qg_best = [r for r in paired if r["adaptive_method"] == "QGCUDA" and r["fixed_method"] == "FIXED_65536"]
    qg_ratio = statistics.median(float(r["adaptive_over_fixed_wall_ratio"]) for r in qg_best)
    qg_wins = sum(float(r["adaptive_wall_s"]) < float(r["fixed_wall_s"]) for r in qg_best)
    qg_sse_delta = statistics.median(float(r["adaptive_minus_fixed_sse"]) for r in qg_best)
    phase_b_rows = read_csv(DECAY / "phase0_convergence_diagnostic_retry1" / "individual_runs.csv")
    differences = []
    for seed in (0, 1, 2):
        cumul = next(r for r in phase_b_rows if r["method"] == "CUMULATIVE" and int(r["seed"]) == seed)
        decay = next(r for r in phase_b_rows if r["method"] == "DECAYED_FIXED_8192" and int(r["seed"]) == seed)
        differences.append(float(decay["final_exact_sse"]) - float(cumul["final_exact_sse"]))
    v2_fixed = [r for r in read_csv(DECAY / "final_summary" / "paired_v2_vs_fixed.csv")
                if r["fixed_method"] == "FIXED_DECAY_B65536"]
    v2_ratio = statistics.median(float(r["paired_v2_over_fixed_wall_ratio"]) for r in v2_fixed)
    v2_gap_delta = statistics.median(float(r["v2_minus_fixed_sse_gap_percentage_points"]) for r in v2_fixed)
    v2_lower = sum(float(r["v2_final_sse"]) < float(r["fixed_final_sse"]) for r in v2_fixed)
    v2_hits = sum(r["target_hit"] == "True" for r in c_rows if r["method"] == "ADAPTIVE_V2")
    fixed_hits = sum(r["target_hit"] == "True" for r in c_rows if r["method"] == "FIXED_DECAY_B65536")
    c_g0 = next(r for r in c_methods if r["method"] == "G0_EXACT_CUDA")
    c_v2 = next(r for r in c_methods if r["method"] == "ADAPTIVE_V2")
    c_bestfixed = next(r for r in c_methods if r["method"] == "FIXED_DECAY_B65536")
    a_best = next(r for r in a_methods if r["method"] == "FIXED_65536")
    a_qg = next(r for r in a_methods if r["method"] == "QGCUDA")
    phase_b_final = [r for r in b["runs"]]
    return f"""# Research findings: adaptive optimization follow-up

## Scope and evidence policy

This package consolidates prior authorized, completed experiments rather than rerunning them. It preserves and copies their raw evidence unchanged. No algorithms, experiment outcomes, quality targets, manuscript or input imagery were modified. Real-data runs executed on the RTX 4050 with CUDA simulation disabled. Main sources are the controlled 105-fit cumulative sweep and the 90-fit decayed validation study; the matched three-seed cumulative/decay trace is diagnostic-only.

Validation data are Sentinel-2 Level-2A 31UFU, crop N=262,144, D=4, K=16. Development data are 31UFS at the same shape. The 5% evaluation target remains the existing seed-specific exact-CPU-reference SSE ×1.05; it is evaluation-only. Exact CUDA G0 is the principal runtime/quality comparator. No target value was passed to a controller.

## Protocol audit

**Phase A:** The saved seven-arm study has {a['runs']} fits (five seeds × three measured repetitions per method) and two warm-ups per method. Fixed widths, QGCUDA and G3CUDA use exact assignment, cumulative float64 sums and int64 counts, the same disjoint 1,024-point probe/training complement, shared per-seed/repetition C0, 40.96M observations and common 8.192M checkpoints. Each batch is uniform without replacement from the same eligible population. Batch-width-specific RNG calls produce different point sequences; the archive does not contain per-batch hashes for this older study, so exact sampled-sequence equality is not claimed. Target is evaluator-only. G0 is exact CUDA Lloyd and is not matched by Mini-Batch observation count.

**Phase B:** The three-seed trace uses identical C0, X, 40.96M observations, batch width and per-update sampled-index hashes between cumulative and decayed arms. It is deliberately marked diagnostic-only because trace instrumentation copies state and changes timing.

**Phase C:** The saved development pilot preceded a frozen validation protocol; validation comprises {c['runs']} fits (five seeds × three repetitions per method) with identical seed-specific C0, common 40.96M budget, five shared checkpoints, exact assignment and same observation-scaled decay rule for all decayed methods. Sample-index hashes are retained per update. Repetitions reuse each seed's deterministic stream, so seed—not repetition—is the independent data/init unit. The exact CUDA comparator terminates by Lloyd convergence, while Mini-Batch arms terminate at the observation budget.

## Phase A — cumulative fixed-size sweep and existing adaptive policies

| Arm | Median wall, s (IQR) | Median final exact SSE | Frozen target hits | Median updates |
|---|---:|---:|---:|---:|
| Fixed B=8,192 | 33.082 (1.947) | 27,607.50 | 0/15 | 5,000 |
| Fixed B=16,384 | 24.230 (1.080) | 27,816.58 | 0/15 | 2,500 |
| Fixed B=32,768 | 16.131 (1.024) | 28,120.75 | 0/15 | 1,250 |
| Fixed B=65,536 | {float(a_best['median_external_wall_s']):.3f} ({float(a_best['wall_iqr_s']):.3f}) | {float(a_best['median_final_exact_sse']):.2f} | {a_best['target_hits']} | {a_best['median_steps']} |
| QGCUDA | {float(a_qg['median_external_wall_s']):.3f} ({float(a_qg['wall_iqr_s']):.3f}) | {float(a_qg['median_final_exact_sse']):.2f} | {a_qg['target_hits']} | {a_qg['median_steps']} |
| G3CUDA | 57.024 (2.894) | 27,603.51 | 0/15 | 9,250 |
| Exact G0 CUDA | 0.713 (0.330) | 25,159.27 | 15/15 | 26 |

Fixed batch 65,536 was the fastest fixed cumulative arm. QGCUDA / fixed-65,536 paired median wall ratio was {qg_ratio:.3f}; QGCUDA was faster in {qg_wins}/{len(qg_best)} pairs, and its median SSE-minus-fixed difference was {qg_sse_delta:.2f}. QGCUDA improved SSE relative to that fixed arm but took longer; neither reached the target. Thus QGCUDA's large advantage over batch 8,192 is not evidence of adaptive benefit over the best fixed batch. G3CUDA was slowest and also missed all targets.

The exact CUDA G0 median fit was 0.713 s and reached the target in all runs. This is a complete-fit comparison, not Mini-Batch time-to-quality at equal work; full Lloyd passes and Mini-Batch presentations are not interchangeable.

## Phase B — cumulative update dynamics and decay

At the 40.96M cap, cumulative SSE was still decreasing at every saved checkpoint, while checkpoint-to-checkpoint centroid displacement contracted but remained nonzero. This supports slow adaptation/inertia rather than a demonstrated complete plateau. Matched final exact SSE differences (decayed minus cumulative) for seeds 0, 1 and 2 were {', '.join(f'{x:,.2f}' for x in differences)}. Decayed updates had lower SSE in all three same-sample sequences. The result supports cumulative-history inertia as a contributor in the tested setup; it does not prove it is the only cause or establish general superiority. Cluster update counts/effective weights and movements are in the copied phase-B checkpoint and trace files.

## Phase C — fixed decay versus Adaptive V2

| Arm | Median complete wall, s (IQR) | Median final exact SSE (IQR) | Median gap to exact CUDA | Target hits | Median updates |
|---|---:|---:|---:|---:|---:|
| Fixed decay B=8,192 | 29.662 (2.214) | 24,703.84 (1,262.64) | 5.928% | 6/15 | 5,000 |
| Fixed decay B=16,384 | 16.005 (1.240) | 24,709.33 (972.23) | 6.110% | 6/15 | 2,500 |
| Fixed decay B=32,768 | 13.064 (8.001) | 24,719.83 (992.84) | 6.577% | 6/15 | 1,250 |
| Fixed decay B=65,536 | {float(c_bestfixed['median_wall_s']):.3f} ({float(c_bestfixed['wall_iqr_s']):.3f}) | {float(c_bestfixed['median_final_exact_sse']):.2f} ({float(c_bestfixed['final_sse_iqr']):.2f}) | {float(c_bestfixed['median_gap_to_exact_cuda_pct']):.3f}% | {c_bestfixed['target_hits']}/15 | {c_bestfixed['median_steps']} |
| Adaptive V2 | {float(c_v2['median_wall_s']):.3f} ({float(c_v2['wall_iqr_s']):.3f}) | {float(c_v2['median_final_exact_sse']):.2f} ({float(c_v2['final_sse_iqr']):.2f}) | {float(c_v2['median_gap_to_exact_cuda_pct']):.3f}% | {c_v2['target_hits']}/15 | {c_v2['median_steps']} |
| Exact G0 CUDA | {float(c_g0['median_wall_s']):.3f} ({float(c_g0['wall_iqr_s']):.3f}) | {float(c_g0['median_final_exact_sse']):.2f} | 0% by reference | {c_g0['target_hits']}/15 | {c_g0['median_steps']} |

The fastest fixed decay was batch 65,536. V2/fixed-65,536 paired median wall ratio was {v2_ratio:.3f}; V2 had lower SSE in {v2_lower}/{len(v2_fixed)} pairs, with paired median gap difference {v2_gap_delta:.3f} percentage points. This exceeds the predeclared 0.5-point quality-comparability band; equal-quality efficiency is not established. Both hit the frozen target in {v2_hits}/15 and {fixed_hits}/15 fits, respectively. Their observed first target crossings are checkpoint-censored; V2 did not establish a time-to-target advantage. Exact G0 reached target in all 15 fits with median 1.227 s complete-fit time (an upper bound on its uncaptured first-crossing time).

The controller made 45 actual increases over 15 V2 fits, from its initial 8,192 batch to the 65,536 maximum, and logged 81 stagnation no-ops at the bound. The recorded decision reasons were active/progress-or-insufficient-evidence and probe-objective stagnation; no deterioration-triggered decisions were recorded. Since the current V2 already uses fixed-probe objective, movement and label-change signals, and the diagnostic shows no harmful probe deterioration to motivate rollback, a new rollback-based quality-aware decayed variant was not implemented. This avoids adding an unsupported controller and preserves V2 as-is.

The optional schedule replay was skipped: the preregistered V2 advantage gate over the fastest fixed decay failed. Therefore no feedback-specific benefit can be separated from batch scheduling.

## Phase D — performance evidence and optimization decision

The cumulative timing rows show the largest recorded groups are batch assignment and accumulation/update. For example, median B=8,192 values were 15.159 s assignment and 12.971 s accumulation/update; at B=65,536 they were 5.018 s and 2.178 s. QGCUDA recorded 5.777 s assignment and 3.037 s update; G3CUDA 26.865 s and 23.075 s. Decayed B=8,192 recorded 13.408 s assignment and 12.887 s update; B=65,536 recorded 4.213 s and 2.051 s. These are instrumented grouped intervals—not isolated kernel durations—and may include host launch, synchronization and transfer costs. Do not sum them to reconstruct external wall time.

The code keeps GPU data/state buffers persistent across iterations, but uploads sampled point indices and synchronizes/reduces at several update/probe boundaries. The earlier Nsight Systems attempt generated no usable trace due to an importer failure. Thus per-kernel launch gaps, individual H2D/D2H transfers, synchronization wait and memory traffic were not separately measured. No low-level optimization was justified or applied. The evidence is consistent with many small batches paying more repeated assignment/update and host sampling/launch overhead, while larger batches reduce update count; it cannot attribute a precise fraction to atomics, transfer or kernel-launch latency.

## Answers and research conclusions

1. **Best fixed cumulative batch:** 65,536 by median runtime in this 31UFU, K=16, 40.96M validation study; it missed the frozen target in every run.
2. **QGCUDA versus best fixed:** QGCUDA did not win runtime against B=65,536; it had lower final SSE but still missed the target.
3. **Cumulative quality gap:** cumulative SSE continued to improve slowly to the cap. A matched three-seed diagnostic found lower SSE with observation-scaled decay on identical samples, making cumulative-history inertia a plausible contributor but not the sole proven cause.
4. **Decay:** improved final SSE in all three matched cumulative-vs-decayed diagnostic seeds and produced strong validation SSE relative to cumulative methods. This is evidence for the tested scene/configuration, not universal optimality.
5. **Adaptive V2 versus best fixed decay:** V2 was slower (ratio {v2_ratio:.3f}) and failed the preregistered quality-equivalence band; no efficiency advantage is established.
6. **Quality-aware adaptation:** no new rollback variant was justified by observed controller signals; rollback benefit remains untested. Existing V2 used objective trend signals but did not show advantage over best fixed.
7. **GPU bottlenecks:** grouped timings point to assignment and accumulation/update as large measured categories; individual kernel/transfer attribution is unavailable.
8. **Sources of observed improvements:** larger batches reduce update count and runtime; decayed update statistics improve SSE in the matched diagnostic; adaptive scheduling improved over smaller fixed batches in some comparisons but did not beat the best fixed batch. No low-level optimization was tested.
9. **Compared with exact CUDA:** exact G0 was much faster and met the frozen target across this study; Mini-Batch time-to-quality superiority is unsupported.
10. **Defensible contribution:** a controlled real-Sentinel-2 comparison showing workload-specific exact CUDA results, batch-size/runtime trade-offs, a matched diagnostic supporting a decay benefit, and negative evidence that the tested adaptive controllers did not beat the best fixed policies at comparable quality.

## Skips, failures and limitations

- The already completed 105-fit matrix was reused instead of repeating it. A prior three-seed 40M QG pilot is retained as supplemental evidence, but it did not include all four fixed widths; no new all-arm sequential pilot was run.
- No schedule replay: predeclared benefit gate failed.
- No new quality-aware-decay implementation: no deterioration-triggered V2 events were observed to motivate rollback; controller inefficiency versus best fixed remains unresolved.
- No GPU kernel optimization or fresh profiler capture: prior Nsight attempt yielded no trace. Grouped timing fields cannot replace profiler evidence.
- One development and one validation crop, D=4,K=16; target hits and SSE are not semantic land-cover accuracy. Five seeds are the independent units; repetitions share per-seed streams. Time-to-target is checkpoint-censored for Mini-Batch.
- In Phase A, sample eligibility and deterministic sampling policy are shared, but batch-index hashes were not recorded. Per-batch sequence identity is unavailable. Phase C includes index hashes.

## Evidence paths

- Cumulative fixed/QG/G3/G0 raw matrix: `phaseA_fixed_batch_sweep/source_archive/individual_runs.csv`.
- Cumulative checkpoints, controller events and timing: same archive, `checkpoints.csv`, `controller_events.csv`, `timing_breakdown.csv`.
- Matched cumulative/decay traces and cluster counts: `phaseB_quality_diagnostics/matched_cumulative_vs_decay_trace/`.
- Fixed-decay/V2 validation raw fits, checkpoints, hashes and controller events: `phaseC_adaptive_decay/validation_31UFU/`.
- GPU timing/profiling limits: `phaseD_gpu_performance/`.
- Figures: `figures/`.
- All copied and derived files: root `checksums.sha256`; manifest/index: root `experiment_index.json`.
"""


def _render_readme(figures: list[str]) -> str:
    return f"""# PCAP adaptive optimization follow-up

This dated package is a verified consolidation of already completed studies. It includes no duplicate clustering fits and no SAFE imagery. See `final_summary/final_findings.md` for methods, results, negative outcomes, skips and limitations. Raw study archives are copied under the Phase A–C folders without altering their source bytes. `experiment_index.json` identifies original archives and their provenance. `checksums.sha256` covers every file in this package except the checksum file itself.

## Study sources

- Phase A: `phaseA_fixed_batch_sweep/source_archive/` — 105 cumulative fixed/QG/G3/G0 real-GPU fits.
- Phase B: `phaseB_quality_diagnostics/matched_cumulative_vs_decay_trace/` — matched three-seed diagnostic.
- Phase C: `phaseC_adaptive_decay/validation_31UFU/` and `development_pilot_31UFS/` — 90 validation fits and prior development pilot.
- Phase D: `phaseD_gpu_performance/` — grouped timing and profiler limitation record.

## Figures generated from saved measured rows

{chr(10).join('- `figures/' + name + '`' for name in figures)}

No plot is produced for the skipped quality-aware-decay arm or schedule replay.

## Reproduction

The original full-run commands and environment snapshots are preserved in each copied source archive README/manifest. To regenerate only the consolidated report and plots from verified archives, run from repository root:

```bash
.venv/bin/python experiments/finalize_optimization_followup.py
```

The script refuses to overwrite an existing output directory. It does not import or call a clustering fit function.
"""


if __name__ == "__main__":
    print(build_archive())
