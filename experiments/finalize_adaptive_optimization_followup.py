"""Verify and consolidate previously completed PCAP optimization experiments.

This script performs no clustering fits. It copies immutable result archives,
validates their protocol/record counts, and derives report tables and figures.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/20261027-adaptive-optimization-followup"
QG = ROOT / "outputs/20261027-qg-batchsize-ablation"
DECAY = ROOT / "outputs/20261027-decay-memory-study"
FOUR = ROOT / "outputs/20261027-adaptive-four-phase-study"
OLD_DECAY = ROOT / "outputs/20261026-validation-decayed-count-k16-b8192-seed0"
PROFILE = Path.home() / "pcap-profiling-archive/cuda-baseline-N32768-20261020"
A_METHODS = {"FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536", "QGCUDA", "G3CUDA", "G0"}
C_METHODS = {"FIXED_DECAY_B8192", "FIXED_DECAY_B16384", "FIXED_DECAY_B32768", "FIXED_DECAY_B65536", "ADAPTIVE_V2", "G0_EXACT_CUDA"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    # Batch schedules are serialized into one CSV cell for the legacy Phase A
    # runner and can exceed Python csv's conservative 128 KiB default.
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_checksums(directory: Path) -> int:
    lines = (directory / "checksums.sha256").read_text().splitlines()
    total = 0
    for line in lines:
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        path = directory / rel.lstrip("* ")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        total += 1
    if total == 0:
        raise ValueError(f"No checksums in {directory}")
    return total


def validate_sources() -> tuple[dict, dict, dict]:
    am = json.loads((QG / "manifest.json").read_text())
    ac = json.loads((QG / "config.json").read_text())
    ar = rows(QG / "individual_runs.csv")
    if not am.get("complete") or am.get("failures") or len(ar) != 105 or {x["method"] for x in ar} != A_METHODS:
        raise ValueError("Cumulative source is not a complete 105-fit seven-method study")
    cnt = {(m, s): 0 for m in A_METHODS for s in range(5)}
    c0 = {}
    for r in ar:
        cnt[(r["method"], int(r["seed"]))] += 1
        c0.setdefault((r["seed"], r["repeat"]), set()).add(r["C0_sha256"])
        if r["X_sha256"] != ac["x_sha256"]:
            raise ValueError("Cumulative source contains an unexpected feature matrix")
        if r["method"] != "G0" and (int(r["processed_observations"]) != 40_960_000 or r["termination_reason"] != "observation_budget"):
            raise ValueError("Cumulative Mini-Batch source has a mismatched budget/termination")
    if any(v != 3 for v in cnt.values()) or any(len(v) != 1 for v in c0.values()):
        raise ValueError("Cumulative source seed/repetition balance or C0 pairing failed")
    if not ac["disjoint_probe"] or ac["probe_size"] != 1024 or ac["observation_budget"] != 40_960_000:
        raise ValueError("Cumulative controlled-probe protocol mismatch")
    verify_checksums(QG)

    cd = DECAY / "phase0_convergence_diagnostic_retry1"
    br, bt, bc = rows(cd / "individual_runs.csv"), rows(cd / "update_trace.csv"), rows(cd / "checkpoints.csv")
    if len(br) != 6 or len(bc) != 30 or any(r["sampling_stream_matches"] != "True" for r in br):
        raise ValueError("Matched cumulative/decay diagnostic is incomplete")
    h = {}
    for r in bt:
        h.setdefault((r["seed"], r["method"]), []).append(r["sample_indices_sha256"])
    for s in ("0", "1", "2"):
        if h[(s, "CUMULATIVE")] != h[(s, "DECAYED_FIXED_8192")]:
            raise ValueError(f"Matched diagnostic sample hashes differ for seed {s}")

    cm = json.loads((DECAY / "phase2_validation/manifest.json").read_text())
    cc = json.loads((DECAY / "phase2_validation/config.json").read_text())
    cr = rows(DECAY / "phase2_validation/individual_runs.csv")
    if not cm.get("complete") or cm.get("failures") or len(cr) != 90 or {r["method"] for r in cr} != C_METHODS:
        raise ValueError("Decayed validation source is not the complete 90-fit matrix")
    cnt = {(m, s): 0 for m in C_METHODS for s in range(5)}
    c0, x_hashes = {}, set()
    for r in cr:
        cnt[(r["method"], int(r["seed"]))] += 1
        c0.setdefault((r["seed"], r["repeat"]), set()).add(r["C0_sha256"])
        x_hashes.add(r["X_sha256"])
        if r["target_evaluation_only"] != "True":
            raise ValueError("Frozen target is not marked evaluation-only in decayed runs")
        if r["method"] != "G0_EXACT_CUDA" and (int(r["observations_processed"]) != 40_960_000 or r["termination_reason"] != "observation_budget"):
            raise ValueError("Decayed Mini-Batch source has a mismatched budget/termination")
    if any(v != 3 for v in cnt.values()) or any(len(v) != 1 for v in c0.values()) or len(x_hashes) != 1:
        raise ValueError("Decayed validation seed balance, X or C0 pairing failed")
    if cc["observation_budget"] != 40_960_000 or cc["seeds"] != [0, 1, 2, 3, 4]:
        raise ValueError("Decayed validation protocol mismatch")
    verify_checksums(DECAY)
    return {"manifest": am, "config": ac, "runs": ar}, {"runs": br, "trace": bt, "checkpoints": bc}, {"manifest": cm, "config": cc, "runs": cr}


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(data)


def median_iqr(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def summarize_timing(rows_in: list[dict[str, str]], methods: set[str], wall_key: str,
                     field_map: dict[str, str], method_key: str = "method") -> list[dict]:
    """Summarize wall and internal fields independently; never add categories."""
    output = []
    for method in sorted(methods):
        subset = [r for r in rows_in if r[method_key] == method]
        if not subset:
            continue
        med, spread = median_iqr([float(r[wall_key]) for r in subset])
        item = {"method": method, "n_runs": len(subset),
                "external_wall_median_s": med, "external_wall_iqr_s": spread}
        for label, source in field_map.items():
            vals = [float(r[source]) for r in subset if r.get(source) not in (None, "")]
            if vals:
                item[label + "_median_s"], item[label + "_iqr_s"] = median_iqr(vals)
            else:
                item[label + "_median_s"] = ""
                item[label + "_iqr_s"] = ""
        output.append(item)
    return output


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, copy_function=shutil.copy2)


def make_figures(ar: list[dict], acheck: list[dict], cr: list[dict], timing: list[dict], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    names = []
    fixed = ["FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536"]
    xs = [8192, 16384, 32768, 65536]
    # Runtime and SSE versus batch width with run-level scatter and medians.
    for filename, field, ylabel in (("01_runtime_vs_fixed_batch.png", "external_wall_s", "Complete wall time (s)"),
                                    ("02_sse_vs_fixed_batch.png", "final_exact_sse", "Final exact full-data SSE")):
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for b, m in zip(xs, fixed):
            vv = [float(r[field]) for r in ar if r["method"] == m]
            ax.scatter([b]*len(vv), vv, alpha=.45, s=20)
            ax.plot([b], [np.median(vv)], marker="D", color="black", ms=4)
        ad = [float(r[field]) for r in ar if r["method"] == "QGCUDA"]
        ax.axhline(np.median(ad), color="tab:orange", ls="--", label="QGCUDA median")
        ax.set_xscale("log", base=2); ax.set_xticks(xs, [f"{b:,}" for b in xs])
        ax.set_xlabel("Fixed batch size"); ax.set_ylabel(ylabel); ax.set_title("31UFU cumulative Mini-Batch; 40.96M observations")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / filename, dpi=200); plt.close(fig); names.append(filename)

    # Paired QG vs fastest fixed arm.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    q={(r["seed"],r["repeat"]):r for r in ar if r["method"]=="QGCUDA"}
    f={(r["seed"],r["repeat"]):r for r in ar if r["method"]=="FIXED_65536"}
    for k in q:
        ax.plot([float(f[k]["external_wall_s"]),float(q[k]["external_wall_s"])],[float(f[k]["final_exact_sse"]),float(q[k]["final_exact_sse"])],color=".75",lw=.8)
    ax.scatter([float(r["external_wall_s"]) for r in f.values()],[float(r["final_exact_sse"]) for r in f.values()],label="Fixed 65,536")
    ax.scatter([float(r["external_wall_s"]) for r in q.values()],[float(r["final_exact_sse"]) for r in q.values()],label="QGCUDA")
    ax.set(xlabel="Complete fit time (s; lower is better)",ylabel="Final exact SSE (lower is better)",title="Paired QGCUDA versus best fixed batch")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/"03_qg_vs_best_fixed.png",dpi=200);plt.close(fig);names.append("03_qg_vs_best_fixed.png")

    fig,ax=plt.subplots(figsize=(9,5.2))
    for m in [*fixed,"QGCUDA","G3CUDA"]:
        sub=[r for r in acheck if r["method"]==m]
        grp={}
        for r in sub:grp.setdefault(int(r["checkpoint_index"]),[]).append(float(r["full_exact_sse"]))
        kk=sorted(grp)
        if kk:ax.plot([np.median([int(r["processed_observations"]) for r in sub if int(r["checkpoint_index"])==k])/1e6 for k in kk],[np.median(grp[k]) for k in kk],marker="o",ms=3,label=m)
    ax.set(xlabel="Processed observations (millions)",ylabel="Exact full-data SSE",title="Cumulative methods: common checkpoints")
    ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(out/"04_sse_vs_observations.png",dpi=200);plt.close(fig);names.append("04_sse_vs_observations.png")

    ex=next((r for r in ar if r["method"]=="QGCUDA" and r["seed"]=="0" and r["repeat"]=="0"),None)
    if ex:
        schedule=json.loads(ex["batch_schedule"]);fig,ax=plt.subplots(figsize=(8.3,4.7))
        ax.step([int(r["points_processed"])/1e6 for r in schedule],[int(r["actual_batch_size"]) for r in schedule],where="post")
        ax.set(xlabel="Processed observations (millions)",ylabel="Batch size",title="QGCUDA schedule, seed 0 / repetition 0")
        ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"05_qg_batch_trajectory.png",dpi=200);plt.close(fig);names.append("05_qg_batch_trajectory.png")

    crn=[dict(r,wall=float(r["wall_s"])) for r in cr]
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.6))
    for ax,key,label in ((axs[0],"wall","Complete wall time (s)"),(axs[1],"final_exact_sse","Final exact SSE")):
        for i,m in enumerate(("FIXED_DECAY_B65536","ADAPTIVE_V2")):
            v=[float(r[key]) for r in crn if r["method"]==m];ax.scatter([i]*len(v),v,label=m)
        ax.set_xticks([0,1],["Fixed B=65,536","Adaptive V2"]);ax.set_ylabel(label);ax.grid(axis="y",alpha=.25)
    axs[0].legend(fontsize=8);fig.suptitle("31UFU decayed-count family; equal 40.96M budget")
    fig.tight_layout();fig.savefig(out/"06_fixed_decay_vs_v2.png",dpi=200);plt.close(fig);names.append("06_fixed_decay_vs_v2.png")

    # Grouped internal categories are shown individually and are not stacked.
    methods=["FIXED_8192","FIXED_65536","QGCUDA","G3CUDA"];cats=[("sampling_s","Sampling"),("batch_assignment_s","Assignment"),("accumulation_update_s","Update"),("monitoring_s","Monitoring")]
    lookup={r["method"]:r for r in timing};fig,ax=plt.subplots(figsize=(9,4.8));w=.19;pos=np.arange(len(methods))
    for j,(key,label) in enumerate(cats):ax.bar(pos+(j-1.5)*w,[float(lookup[m][key+"_median"]) for m in methods],w,label=label)
    ax.set_xticks(pos,methods);ax.set_ylabel("Median recorded internal interval (s)");ax.set_title("Grouped timing categories; overlap/transfer attribution limitations apply")
    ax.grid(axis="y",alpha=.25);ax.legend(ncol=2);fig.tight_layout();fig.savefig(out/"07_grouped_gpu_timing.png",dpi=200);plt.close(fig);names.append("07_grouped_gpu_timing.png")

    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for ax,rr,ms,title in ((axs[0],ar,sorted(A_METHODS),"Cumulative family"),(axs[1],cr,sorted(C_METHODS),"Decayed family")):
        rate=[sum(r["target_hit"]=="True" for r in rr if r["method"]==m)/max(1,sum(r["method"]==m for r in rr)) for m in ms]
        ax.bar(np.arange(len(ms)),rate);ax.set_xticks(np.arange(len(ms)),ms,rotation=55,ha="right",fontsize=7);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel("Frozen target hit rate");ax.grid(axis="y",alpha=.25)
    fig.suptitle("Evaluation target; all misses remain shown");fig.tight_layout();fig.savefig(out/"08_target_hit_rates.png",dpi=200);plt.close(fig);names.append("08_target_hit_rates.png")

    fig,ax=plt.subplots(figsize=(9.5,4.8));ms=["FIXED_8192","FIXED_16384","FIXED_32768","FIXED_65536","QGCUDA","G3CUDA","G0"]
    for i,m in enumerate(ms):
        hits=[float(r["first_target_crossing_time_s"]) for r in ar if r["method"]==m and r.get("first_target_crossing_time_s") not in ("",None)]
        if hits:ax.scatter([i]*len(hits),hits,s=18)
        else:ax.text(i,.03,"miss",rotation=90,ha="center",transform=ax.get_xaxis_transform(),color="tab:red")
    ax.set_xticks(range(len(ms)),ms,rotation=35,ha="right");ax.set_ylabel("Observed first target crossing time (s)");ax.set_title("Cumulative-family time-to-quality; misses not imputed")
    ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/"09_time_to_quality.png",dpi=200);plt.close(fig);names.append("09_time_to_quality.png")
    return names


def build() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    a,b,c=validate_sources()
    for name in ("phaseA_fixed_batch_sweep","phaseB_quality_diagnostics","phaseC_adaptive_decay","phaseD_gpu_performance","figures","final_summary"):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    copy_tree(QG,OUT/"phaseA_fixed_batch_sweep"/"source_archive")
    copy_tree(DECAY/"phase0_convergence_diagnostic_retry1",OUT/"phaseB_quality_diagnostics"/"matched_cumulative_vs_decay_trace")
    if OLD_DECAY.is_dir():copy_tree(OLD_DECAY,OUT/"phaseB_quality_diagnostics"/"historical_5000_update_single_seed_diagnostic")
    copy_tree(DECAY/"phase1_development_retry1",OUT/"phaseC_adaptive_decay"/"development_pilot_31UFS")
    copy_tree(DECAY/"phase2_validation",OUT/"phaseC_adaptive_decay"/"validation_31UFU")
    shutil.copy2(DECAY/"frozen_validation_protocol.json",OUT/"phaseC_adaptive_decay/frozen_validation_protocol.json")
    (OUT/"phaseC_adaptive_decay/derived").mkdir(parents=True,exist_ok=True)
    for n in ("summary.md","method_summary.csv","paired_v2_vs_fixed.csv","timing_summary.csv","controller_summary.json","cumulative_vs_decay_diagnostic_summary.csv","source_and_measurement_corrections.json","individual_runs_reconciled.csv"):
        p=DECAY/"final_summary"/n
        if p.is_file():shutil.copy2(p,OUT/"phaseC_adaptive_decay/derived"/n)
    pilot=FOUR/"phase3_quality_guarded_adaptive/pilot_40m_disjoint_probe"
    if pilot.is_dir():copy_tree(pilot,OUT/"phaseA_fixed_batch_sweep/prior_quality_guard_pilot")
    for src,dst in ((QG/"timing_breakdown.csv",OUT/"phaseD_gpu_performance/cumulative_timing_breakdown.csv"),(DECAY/"phase2_validation/timing_breakdown.csv",OUT/"phaseD_gpu_performance/decayed_timing_breakdown.csv")):
        shutil.copy2(src,dst)
    if PROFILE.is_dir():copy_tree(PROFILE,OUT/"phaseD_gpu_performance/prior_nsight_attempt")

    # Derived timing table for cumulative methods.
    tr=rows(QG/"timing_breakdown.csv"); timing=[]
    for m in sorted(A_METHODS):
        rr=[r for r in tr if r["method"]==m]
        if not rr:continue
        item={"method":m,"n_runs":len(rr)}
        for field in ("external_wall_s","sampling_s","batch_assignment_s","accumulation_update_s","monitoring_s","controller_s","quality_evaluation_s","final_assignment_s"):
            vals=[float(r[field]) for r in rr if r.get(field) not in (None,"")]
            if vals:item[field+"_median"]=float(np.median(vals));item[field+"_iqr"]=float(np.percentile(vals,75)-np.percentile(vals,25))
            else:item[field+"_median"]="";item[field+"_iqr"]=""
        timing.append(item)
    write_csv(OUT/"final_summary/cumulative_timing_summary.csv",timing)
    # Compare each cumulative-family fit with its paired exact-GPU reference.
    # This is distinct from the frozen CPU-reference evaluation target stored
    # in the original records.
    g0_by_pair={(r["seed"],r["repeat"]):float(r["final_exact_sse"])
                for r in a["runs"] if r["method"]=="G0"}
    gap_rows=[]
    gap_summary={}
    for method in sorted(A_METHODS):
        paired=[]
        for r in a["runs"]:
            if r["method"]!=method:
                continue
            g0=g0_by_pair[(r["seed"],r["repeat"])]
            gap=(float(r["final_exact_sse"])/g0-1.0)*100.0
            paired.append(gap)
            gap_rows.append({"method":method,"seed":r["seed"],"repeat":r["repeat"],
                             "relative_sse_gap_to_paired_cuda_g0_pct":gap})
        if paired:
            med=float(np.median(paired))
            iqr=float(np.percentile(paired,75)-np.percentile(paired,25))
            gap_summary[method]=f"{med:.3f} [{iqr:.3f}]"
    write_csv(OUT/"final_summary/cumulative_gap_to_paired_cuda_g0.csv",gap_rows)
    cr=[dict(r,external_wall_s=r["wall_s"]) for r in c["runs"]]
    figures=make_figures(a["runs"],rows(QG/"checkpoints.csv"),cr,timing,OUT/"figures")
    for src,dst in ((QG/"method_summary.csv",OUT/"final_summary/cumulative_method_summary.csv"),(QG/"paired_comparisons.csv",OUT/"final_summary/cumulative_paired_comparisons.csv"),(DECAY/"final_summary/method_summary.csv",OUT/"final_summary/decayed_method_summary.csv"),(DECAY/"final_summary/paired_v2_vs_fixed.csv",OUT/"final_summary/decayed_v2_paired_comparisons.csv"),(DECAY/"final_summary/cumulative_vs_decay_diagnostic_summary.csv",OUT/"final_summary/cumulative_decay_matched_diagnostic.csv")):
        shutil.copy2(src,dst)
    report=f'''# PCAP adaptive optimization follow-up: results and limitations

## Data and comparison rules

The package consolidates completed records rather than duplicating runs: Phase A has {len(a['runs'])} controlled real-GPU fits, Phase B has {len(b['runs'])} matched diagnostic fits, and Phase C has {len(c['runs'])} validation fits. Data are real Sentinel-2 Level-2A crops: 31UFU validation, N=262,144, D=4, K=16; 31UFS development was used for the decay pilot. The prepared X and per-seed initial-centroid hashes are preserved in manifests. The frozen quality threshold remains the exact-CPU-reference SSE ×1.05 and is evaluation-only. Exact CUDA G0 is the practical reference; Lloyd update-point presentations are not equated to Mini-Batch observations.

Phase A uses the same cumulative update family (float64 sums, int64 counts, empty clusters retain centroids), exact assignment, disjoint fixed 1,024-row probe, training eligibility, observation budget and checkpoints for fixed 8,192/16,384/32,768/65,536, QGCUDA and G3CUDA. Methods share seed/repetition C0. Each batch is sampled without replacement from the same eligible population using the same seeded policy; widths consume different RNG calls, so pointwise-identical sequences are not claimed and batch-level index hashes were not saved in this older matrix. It includes 5 seeds ×3 measured repetitions and two warmups. This already-completed full matrix was reused; no duplicate 3-seed all-arm pilot was run. An earlier 3-seed QG policy pilot is retained separately.

Phase B compares cumulative versus observation-scaled decay at B=8,192, with identical X, C0 and per-update sampled-index hashes for seeds 0–2, 40.96M observations and five checkpoints. It is diagnostic-only: trace copies add overhead, so its time fields are not performance measurements.

Phase C was preceded by a 31UFS development pilot and a frozen validation protocol. It compares fixed decay widths and Adaptive V2 with common seeds, C0, exact GPU assignment, decay rule and 40.96M observation budget on 31UFU: 5 seeds ×3 measured repetitions, two warmups and five checkpoints. Per-update sample hashes and histories are saved. The three repetitions reuse each seed's deterministic sample stream; seeds are the independent data/init units.

## Phase A: cumulative Mini-Batch sweep

| Method | Median complete wall (s) [IQR] | Median final exact SSE | Gap to paired CUDA G0 (%) [IQR] | Frozen target hits | Median updates |
|---|---:|---:|---:|---:|---:|
| Fixed B=8,192 | 33.082 [1.947] | 27,607.50 | {gap_summary['FIXED_8192']} | 0/15 | 5,000 |
| Fixed B=16,384 | 24.230 [1.080] | 27,816.58 | {gap_summary['FIXED_16384']} | 0/15 | 2,500 |
| Fixed B=32,768 | 16.131 [1.024] | 28,120.75 | {gap_summary['FIXED_32768']} | 0/15 | 1,250 |
| Fixed B=65,536 | 9.076 [0.385] | 28,201.46 | {gap_summary['FIXED_65536']} | 0/15 | 625 |
| QGCUDA | 11.132 [1.133] | 27,597.39 | {gap_summary['QGCUDA']} | 0/15 | 879 |
| G3CUDA | 57.024 [2.894] | 27,603.51 | {gap_summary['G3CUDA']} | 0/15 | 9,250 |
| Exact G0 CUDA | 0.713 [0.330] | 25,159.27 | {gap_summary['G0']} | 15/15 | 26 |

Gap to paired CUDA G0 is a descriptive relative SSE comparison; frozen target hits use the separate exact-CPU-reference threshold ×1.05. Fixed B=65,536 was the fastest fixed width. The ratio of group medians, QGCUDA/fixed B=65,536, was 1.226; the median of paired per-seed/repetition ratios was 1.239, with QGCUDA faster in 0/15 pairs. QGCUDA yielded median SSE 675.56 lower, but neither method reached the frozen target. Thus the QG runtime advantage over B=8,192 does not survive comparison with the best fixed size. G3CUDA was slowest and missed all targets. G0 was much faster and met the target, but this is not a matched work budget; full Lloyd updates and Mini-Batch presentations differ.

## Phase B: cumulative update quality

At the 40.96M cap, cumulative SSE kept decreasing at every shared checkpoint, while maximum centroid movement contracted from 1.113 to 0.0267 (seed 0; 41.7×), 1.215 to 0.0595 (seed 1; 20.4×), and 0.927 to 0.0469 (seed 2; 19.8×). Movement remained nonzero, so this was slow progress rather than a demonstrated full-data plateau. At the final checkpoint, cumulative cluster update-count max/min ratios were 14.84, 17.11 and 13.55 for seeds 0–2; decayed effective-count ratios were 17.40, 15.95 and 14.82. Every cluster had positive cumulative and effective mass; empty clusters do not explain this gap. Matched final decayed-minus-cumulative SSE was −3,572.30, −1,879.53 and −2,058.28 for seeds 0, 1 and 2. Decayed SSE was lower for all three identical-sample runs. This supports cumulative-history inertia as one contributor in this setup, not the sole general cause. Checkpoint trajectories and count vectors are preserved in the trace files.

## Phase C: decayed fixed batches versus Adaptive V2

| Method | Median wall (s) [IQR] | Median exact SSE [IQR] | Gap to exact CUDA | Hits | Median updates |
|---|---:|---:|---:|---:|---:|
| Fixed decay B=8,192 | 29.662 [2.214] | 24,703.84 [1,262.64] | 5.928% | 6/15 | 5,000 |
| Fixed decay B=16,384 | 16.005 [1.240] | 24,709.33 [972.23] | 6.110% | 6/15 | 2,500 |
| Fixed decay B=32,768 | 13.064 [8.001] | 24,719.83 [992.84] | 6.577% | 6/15 | 1,250 |
| Fixed decay B=65,536 | 7.517 [4.706] | 24,764.59 [1,024.25] | 6.744% | 6/15 | 625 |
| Adaptive V2 | 8.903 [5.539] | 24,703.25 [1,262.59] | 5.951% | 6/15 | 1,773 |
| Exact G0 CUDA | 1.227 [0.126] | 24,395.03 [1,738.09] | 0% | 15/15 | 144 |

Adaptive V2 had lower SSE than fixed B=65,536 in 15/15 paired runs, but its paired median gap change was −0.793 percentage points, outside the preregistered ±0.5-point equivalence band. Its median wall ratio to the fastest fixed decay was 1.270 (slower). Both target-hit rates were 6/15. Their median observed checkpoint-crossing times were 12.172 s for V2 and 3.079 s for fixed B=65,536; crossings are checkpoint-censored. All Mini-Batch fits terminated at the observation budget. V2 does not establish adaptive efficiency at equivalent quality. Exact G0 remains the practical speed/quality reference; its reported complete-fit time bounds but does not timestamp its first target crossing.

V2's controller made 45 actual increases across 15 runs and reached 65,536; 81 further stagnation decisions were no-ops at the bound. No deterioration-triggered decisions were recorded. Because V2 already responds to probe objective/movement/label signals and observed deterioration did not occur, a new rollback-based quality-aware decay variant was not justified. The optional schedule replay was skipped because V2 failed the predeclared advantage gate against the fastest fixed arm.

## Decay rule and half-life

Both existing fixed-decay and V2 use `lambda_b = 0.5 ** (b/(10N))`, then `S <- lambda_b*S + batch_sum`, `W <- lambda_b*W + batch_count`, `C=S/W`. For processed observation total M, retained prior-history mass is `0.5 ** (M/(10N))`; the historical-statistic half-life is 10N observations regardless of batch schedule. It is block decay: observations inside one batch get equal weight. At B=65,536, exact sequential within-batch decay would have oldest/newest relative weight 0.98282, so the block approximation's within-batch discrepancy is about 1.72%. No normalization or algorithm semantics were changed.

## Phase D: timing evidence and optimization decision

Available timing instrumentation records grouped host intervals. In cumulative runs, median assignment/update were 15.159/12.971 s at B=8,192; 5.018/2.178 s at B=65,536; QG 5.777/3.037 s; G3 26.865/23.075 s. Decayed B=8,192 records 13.408/12.887 s and B=65,536 4.213/2.051 s. These are not isolated kernel times and may include launches, synchronization and transfer behavior. Sampling and probe monitoring are also recorded. The prior Nsight Systems run produced no usable timeline due to an importer failure. Individual H2D/D2H costs, launch gaps, memory traffic and kernel durations remain unavailable. No low-level optimization was attempted; the evidence suggests larger batches amortize repeated work, but cannot support causal attribution to atomics or launch latency.

## Answers and limitations

1. **Best fixed cumulative size:** 65,536 by median time in this 31UFU/K=16/40.96M study; it missed target 15/15.
2. **QGCUDA:** does not beat that fixed size; its SSE is lower but wall time is higher, and neither meets the target.
3. **Cumulative gap:** progress remained positive at the cap but slowed. The matched diagnostic implicates historical-count inertia; causal generalization is not established.
4. **Decay:** lower final SSE in all three identical-sample diagnostic seeds and competitive 5-seed validation outcomes versus cumulative methods, with no claim of universal superiority.
5. **Adaptive V2:** lower SSE than the best fixed decay, but outside the predeclared quality-equivalence band and slower; no equal-quality efficiency advantage.
6. **Quality-aware adaptation:** no new rollback controller was added; available signals did not show deterioration requiring rollback. Benefit remains untested.
7. **GPU costs:** assignment and accumulation/update are the largest available grouped categories; per-kernel attribution is unavailable.
8. **Observed effects:** larger batches reduce update count and runtime; decay improves tested clustering SSE; adaptive policies beat some small-batch baselines but not best fixed batch; low-level GPU optimization has not been evaluated.
9. **Exact CUDA:** fastest and highest-quality under these runs; Mini-Batch did not reach target in all seeds and did not show better time-to-target.
10. **Defensible contribution:** controlled real-Sentinel-2 evidence on exact CUDA, batch-size/runtime trade-offs, a same-sample decay diagnostic, and a negative result for Adaptive V2 efficiency versus best fixed decay.

Limitations include one 31UFU validation crop, one 31UFS development crop, D=4, K=16, five seeds, repeated deterministic sample streams, checkpoint-censored Mini-Batch target crossings, Phase A without per-batch index hashes, no ground-truth land-cover labels, and no usable per-kernel profiler trace. All capped Mini-Batch runs are reported as budget-terminated, not converged.

## Figures

{chr(10).join(f'- `../figures/{n}`' for n in figures)}

## Source records

- Cumulative raw fits/checkpoints/timing/controllers: `../phaseA_fixed_batch_sweep/source_archive/`.
- Same-sample cumulative/decay trace and cluster diagnostics: `../phaseB_quality_diagnostics/matched_cumulative_vs_decay_trace/`.
- Decay/V2 raw fits, checkpoints, sample hashes, controllers and frozen protocol: `../phaseC_adaptive_decay/`.
- Grouped timing and profiler blocker: `../phaseD_gpu_performance/`.
- Every copied/derived artifact is checksummed in the package-root `checksums.sha256`.
'''
    (OUT/"final_summary/final_findings.md").write_text(report,encoding="utf-8")
    (OUT/"README.md").write_text("""# Adaptive optimization follow-up\n\nA verified consolidation of already completed experiments; it contains no new clustering fits and no SAFE imagery. See `final_summary/final_findings.md`. Source archives are copied into the phase directories with original files unchanged. The root `experiment_index.json` records source archives, methods, validation checks, skips and limitations.\n\nRebuild derived tables and figures from the preserved source records (only if the output directory has first been safely moved/archived; the script refuses overwrite):\n\n```bash\n.venv/bin/python experiments/finalize_adaptive_optimization_followup.py\n```\n\nGrouped timing categories are not per-kernel timings and are not added to create wall time. No figure is fabricated for the skipped schedule replay or quality-aware-decay arm.\n""",encoding="utf-8")
    index={"study":"PCAP adaptive optimization follow-up","created_utc":datetime.now(timezone.utc).isoformat(),"branch":"experiment-adaptive-v2","new_fits_executed":False,"source_validation":{"phaseA_runs":len(a["runs"]),"phaseB_runs":len(b["runs"]),"phaseC_runs":len(c["runs"]),"phaseA_checksum_entries":verify_checksums(QG),"decay_archive_checksum_entries":verify_checksums(DECAY)},"sources":{"phaseA":str(QG.relative_to(ROOT)),"phaseB":str((DECAY/"phase0_convergence_diagnostic_retry1").relative_to(ROOT)),"phaseC":str((DECAY/"phase2_validation").relative_to(ROOT)),"phaseD_profiler":str(PROFILE)},"skipped":{"phaseA_duplicate_pilot":"The already completed 105-fit full matrix was reused; prior QG policy pilot is preserved, but is not mislabeled as the seven-arm sweep pilot.","quality_aware_decay":"Skipped because V2 recorded no deterioration decisions that would motivate rollback; adding a new controller would lack a specific supported hypothesis.","schedule_replay":"Skipped because V2 did not pass the preregistered advantage gate versus best fixed decay.","low_level_gpu_optimization":"Skipped because no usable per-kernel timeline exists; grouped timing cannot identify a safe targeted optimization."},"figures":figures,"source_hashes":{"phaseA_manifest":digest(QG/"manifest.json"),"phaseC_manifest":digest(DECAY/"phase2_validation/manifest.json"),"frozen_protocol":digest(DECAY/"frozen_validation_protocol.json")}}
    (OUT/"experiment_index.json").write_text(json.dumps(index,indent=2)+"\n",encoding="utf-8")
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="checksums.sha256")
    (OUT/"checksums.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files),encoding="utf-8")
    if verify_checksums(OUT)!=len(files):raise RuntimeError("Output checksum audit count mismatch")
    return OUT


if __name__ == "__main__":
    print(build())
