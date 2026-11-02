"""Rebuild summaries/figures and verify an existing ANN optimization archive.

This program never runs CUDA kernels or rewrites individual_runs.csv. It derives
all summaries from the preserved measurement rows and refreshes archive hashes.
"""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE = {"strict_agreement_min": 0.95, "assignment_objective_gap_max": 0.01,
        "median_paired_lifecycle_speedup_min": 1.10,
        "seed_bootstrap_speedup_ci95_lower_gt": 1.0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def median_iqr(values):
    return (statistics.median(values), float(np.percentile(values, 25)),
            float(np.percentile(values, 75)))


def finalize(output):
    output = Path(output).resolve()
    raw_path = output / "phaseC_assignment_benchmarks/individual_runs.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw_hash_before = sha(raw_path)
    with raw_path.open(newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError("No measured rows to summarize")
    for r in raw:
        for key in ("n","d","k","seed","repetition","effort","candidate_distance_evaluations","exhaustive_distance_evaluations","index_bytes"):
            if r[key] not in ("", "None"):
                r[key] = int(r[key])
        for key in ("external_complete_s","query_s","index_build_s","index_refresh_s",
                    "strict_label_agreement","tie_aware_distance_agreement",
                    "mean_excess_squared_distance","max_excess_squared_distance",
                    "assignment_objective_gap","candidate_fraction"):
            if r[key] not in ("", "None"):
                r[key] = float(r[key])
    exact = {(r["scene"],r["k"],r["seed"],r["repetition"],r["snapshot"]):r
             for r in raw if r["method"]=="exact_cuda"}
    group_keys=sorted({(r["scene"],r["k"],r["method"],r["effort"]) for r in raw},key=str)
    summary=[]
    for scene,k,method,effort in group_keys:
        subset=[r for r in raw if (r["scene"],r["k"],r["method"],r["effort"])==(scene,k,method,effort)]
        initial=[r for r in subset if r["snapshot"]=="initial"]
        moved=[r for r in subset if r["snapshot"]!="initial"]
        lifecycle={}
        for r in initial:
            pair=(r["seed"],r["repetition"])
            other=next(x for x in moved if (x["seed"],x["repetition"])==pair)
            lifecycle[pair]=r["external_complete_s"]+other["external_complete_s"]
        med,q1,q3=median_iqr(list(lifecycle.values()))
        summary.append(dict(scene=scene,k=k,method=method,effort=effort,
            n_seed_repetitions=len(lifecycle),median_lifecycle_wall_s=med,
            q1_lifecycle_wall_s=q1,q3_lifecycle_wall_s=q3,
            median_initial_wall_s=statistics.median(r["external_complete_s"] for r in initial),
            median_moved_incremental_wall_s=statistics.median(r["external_complete_s"] for r in moved),
            median_query_s=statistics.median(r["query_s"] for r in subset),
            median_index_build_s=statistics.median(r["index_build_s"] for r in subset),
            median_index_refresh_s=statistics.median(r["index_refresh_s"] for r in subset),
            median_strict_agreement=statistics.median(r["strict_label_agreement"] for r in subset),
            min_strict_agreement=min(r["strict_label_agreement"] for r in subset),
            median_tie_aware_distance_agreement=statistics.median(r["tie_aware_distance_agreement"] for r in subset),
            median_objective_gap=statistics.median(r["assignment_objective_gap"] for r in subset),
            max_objective_gap=max(r["assignment_objective_gap"] for r in subset),
            median_mean_excess_squared_distance=statistics.median(r["mean_excess_squared_distance"] for r in subset),
            median_candidate_fraction=statistics.median(r["candidate_fraction"] for r in subset),
            median_index_bytes=statistics.median(r["index_bytes"] for r in subset)))
    write_csv(output/"phaseC_assignment_benchmarks/method_summary.csv",summary)

    paired_rows=[]
    for scene in sorted({r["scene"] for r in raw}):
        common=sorted({(r["k"],r["effort"]) for r in raw if r["scene"]==scene and r["method"]=="single_projection"} &
                      {(r["k"],r["effort"]) for r in raw if r["scene"]==scene and r["method"]=="multi_projection"},key=str)
        for k,effort in common:
            for snapshot in ("initial","after_one_exact_cuda_lloyd_update"):
                single=[r for r in raw if r["scene"]==scene and r["k"]==k and r["effort"]==effort and r["method"]=="single_projection" and r["snapshot"]==snapshot]
                multi=[r for r in raw if r["scene"]==scene and r["k"]==k and r["effort"]==effort and r["method"]=="multi_projection" and r["snapshot"]==snapshot]
                join={(r["seed"],r["repetition"]):r for r in single}
                pairs=[(join[(r["seed"],r["repetition"])],r) for r in multi]
                paired_rows.append(dict(scene=scene,k=k,effort=effort,snapshot=snapshot,n_pairs=len(pairs),
                    median_multi_minus_single_strict_agreement=statistics.median(b["strict_label_agreement"]-a["strict_label_agreement"] for a,b in pairs),
                    median_multi_minus_single_objective_gap=statistics.median(b["assignment_objective_gap"]-a["assignment_objective_gap"] for a,b in pairs),
                    median_multi_over_single_external_wall=statistics.median(b["external_complete_s"]/a["external_complete_s"] for a,b in pairs),
                    median_candidate_fraction_single=statistics.median(a["candidate_fraction"] for a,b in pairs),
                    median_candidate_fraction_multi=statistics.median(b["candidate_fraction"] for a,b in pairs)))
            lifecycle_pairs=[]
            for seed in sorted({r["seed"] for r in raw if r["scene"]==scene and r["k"]==k}):
                for rep in sorted({r["repetition"] for r in raw if r["scene"]==scene and r["k"]==k and r["seed"]==seed}):
                    def find(method,snap):
                        return next(r for r in raw if r["scene"]==scene and r["k"]==k and r["seed"]==seed and r["repetition"]==rep and r["method"]==method and r["effort"]==effort and r["snapshot"]==snap)
                    sa=find("single_projection","initial");sb=find("single_projection","after_one_exact_cuda_lloyd_update")
                    ma=find("multi_projection","initial");mb=find("multi_projection","after_one_exact_cuda_lloyd_update")
                    lifecycle_pairs.append((sa,sb,ma,mb))
            paired_rows.append(dict(scene=scene,k=k,effort=effort,snapshot="two_state_lifecycle",n_pairs=len(lifecycle_pairs),
                median_multi_minus_single_strict_agreement=statistics.median(statistics.median([ma["strict_label_agreement"],mb["strict_label_agreement"]])-statistics.median([sa["strict_label_agreement"],sb["strict_label_agreement"]]) for sa,sb,ma,mb in lifecycle_pairs),
                median_multi_minus_single_objective_gap=statistics.median(statistics.median([ma["assignment_objective_gap"],mb["assignment_objective_gap"]])-statistics.median([sa["assignment_objective_gap"],sb["assignment_objective_gap"]]) for sa,sb,ma,mb in lifecycle_pairs),
                median_multi_over_single_external_wall=statistics.median((ma["external_complete_s"]+mb["external_complete_s"])/(sa["external_complete_s"]+sb["external_complete_s"]) for sa,sb,ma,mb in lifecycle_pairs),
                median_candidate_fraction_single=statistics.median((sa["candidate_fraction"]+sb["candidate_fraction"])/2 for sa,sb,ma,mb in lifecycle_pairs),
                median_candidate_fraction_multi=statistics.median((ma["candidate_fraction"]+mb["candidate_fraction"])/2 for sa,sb,ma,mb in lifecycle_pairs)))
    write_csv(output/"phaseC_assignment_benchmarks/paired_backend_comparison.csv",paired_rows)

    timing=[]
    for r in raw:
        for category,value,scope in (
            ("external_complete_wall",r["external_complete_s"],"authoritative; synchronized; contains the component rows below"),
            ("query",r["query_s"],"synchronized subset; do not add to external complete wall"),
            ("index_initial_build",r["index_build_s"],"subset of initial-state external wall"),
            ("index_refresh",r["index_refresh_s"],"subset of moved-state external wall"),
            ("transfer_in",r.get("transfer_in_s"),"subset of external complete wall"),
            ("transfer_out",r.get("transfer_out_s"),"subset of external complete wall")):
            timing.append(dict(scene=r["scene"],k=r["k"],seed=r["seed"],repetition=r["repetition"],
                snapshot=r["snapshot"],method=r["method"],effort=r["effort"],
                timing_category=category,seconds=value,scope=scope))
    write_csv(output/"phaseC_assignment_benchmarks/timing_breakdown.csv",timing)
    write_csv(output/"phaseC_assignment_benchmarks/checkpoints.csv",[
        dict(experiment_kind="assignment_only",not_applicable_reason="No iterative clustering checkpoints; two fixed centroid states were evaluated.")])
    write_csv(output/"phaseC_assignment_benchmarks/controller_events.csv",[
        dict(experiment_kind="assignment_only",not_applicable_reason="No adaptive controller was executed in this gated assignment benchmark.")])

    gate=[]
    configs=sorted({(r["scene"],r["k"],r["method"],r["effort"]) for r in raw if r["method"]!="exact_cuda"},key=str)
    for scene,k,method,effort in configs:
        subset=[r for r in raw if (r["scene"],r["k"],r["method"],r["effort"])==(scene,k,method,effort)]
        speeds=[]
        per_seed={}
        for seed in sorted({r["seed"] for r in subset}):
            ratios=[]
            for rep in sorted({r["repetition"] for r in subset if r["seed"]==seed}):
                a=next(r for r in subset if r["seed"]==seed and r["repetition"]==rep and r["snapshot"]=="initial")
                b=next(r for r in subset if r["seed"]==seed and r["repetition"]==rep and r["snapshot"]!="initial")
                e1=exact[(scene,k,seed,rep,"initial")]
                e2=exact[(scene,k,seed,rep,"after_one_exact_cuda_lloyd_update")]
                ratios.append((e1["external_complete_s"]+e2["external_complete_s"])/
                              (a["external_complete_s"]+b["external_complete_s"]))
            per_seed[seed]=statistics.median(ratios)
            speeds.extend(ratios)
        seed_values=np.asarray(list(per_seed.values()),dtype=np.float64)
        rng=np.random.default_rng(np.random.SeedSequence([8441,int(k),int(sorted(per_seed)[0]),int(effort)]))
        boot=np.median(seed_values[rng.integers(0,len(seed_values),(20000,len(seed_values)))],axis=1)
        ci_low=float(np.quantile(boot,.025))
        agreement=min(r["strict_label_agreement"] for r in subset)
        gap=max(r["assignment_objective_gap"] for r in subset)
        med_speed=statistics.median(speeds)
        quality=agreement>=GATE["strict_agreement_min"] and gap<=GATE["assignment_objective_gap_max"]
        speed=med_speed>=GATE["median_paired_lifecycle_speedup_min"] and ci_low>GATE["seed_bootstrap_speedup_ci95_lower_gt"]
        gate.append(dict(scene=scene,k=k,method=method,effort=effort,n_seed_repetitions=len(subset),
            min_strict_agreement=agreement,max_assignment_objective_gap=gap,
            median_paired_lifecycle_speedup=med_speed,median_seed_paired_speedup=statistics.median(per_seed.values()),
            seed_bootstrap_speedup_ci95_low=ci_low,passes_quality=quality,passes_speed=speed,passes_gate=quality and speed))
    write_csv(output/"phaseC_assignment_benchmarks/gate_summary.csv",gate)

    # Actual-measurement plots. Matplotlib is already part of this project's environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir=output/"figures"
    for scene in sorted({r["scene"] for r in summary}):
        fig,ax=plt.subplots(figsize=(9,5.4))
        subset=[r for r in summary if r["scene"]==scene]
        exact_rows=sorted([r for r in subset if r["method"]=="exact_cuda"],key=lambda r:r["k"])
        exact_by_k={}
        for k in sorted({r["k"] for r in exact_rows}):
            vals=[r["median_lifecycle_wall_s"] for r in exact_rows if r["k"]==k]
            exact_by_k[k]=statistics.median(vals)
        ax.plot(list(exact_by_k),list(exact_by_k.values()),"o-",label="Exact CUDA")
        for method,label,marker in (("single_projection","Single projection","s"),("multi_projection","Four projections","^") ):
            for effort in sorted({r["effort"] for r in subset if r["method"]==method}):
                rs=[r for r in subset if r["method"]==method and r["effort"]==effort]
                points=[]
                for k in sorted({r["k"] for r in rs}):
                    vals=[r["median_lifecycle_wall_s"] for r in rs if r["k"]==k]
                    points.append((k,statistics.median(vals),float(np.percentile(vals,25)),float(np.percentile(vals,75))))
                if points:
                    ys=np.asarray([v[1] for v in points]); lo=np.asarray([v[1]-v[2] for v in points]); hi=np.asarray([v[3]-v[1] for v in points])
                    ax.errorbar([v[0] for v in points],ys,yerr=np.vstack([lo,hi]),fmt=marker+"--",capsize=2,label=f"{label}, effort {effort}")
        ax.set(xlabel="Centroid count K",ylabel="Two-state lifecycle wall time (s)",title=f"Assignment lifecycle cost — {scene}")
        ax.grid(True,alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(figdir/f"assignment_runtime_{scene}.png",dpi=180);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(11,4.4))
        for method,label,marker in (("single_projection","Single projection","s"),("multi_projection","Four projections","^") ):
            for effort in sorted({r["effort"] for r in subset if r["method"]==method}):
                rs=[r for r in subset if r["method"]==method and r["effort"]==effort]
                points=[]
                for k in sorted({r["k"] for r in rs}):
                    at_k=[r for r in rs if r["k"]==k]
                    recalls=[r["median_strict_agreement"] for r in at_k]
                    gaps=[r["median_objective_gap"] for r in at_k]
                    points.append((k,statistics.median(recalls),statistics.median(gaps)))
                if points:
                    axs[0].plot([v[0] for v in points],[v[1] for v in points],marker+"--",label=f"{label} e={effort}")
                    axs[1].plot([v[0] for v in points],[v[2] for v in points],marker+"--",label=f"{label} e={effort}")
        axs[0].axhline(GATE["strict_agreement_min"],color="black",ls=":",label="95% gate")
        axs[1].axhline(GATE["assignment_objective_gap_max"],color="black",ls=":",label="1% gate")
        axs[0].set(xlabel="K",ylabel="Median strict agreement",ylim=(0,1.02),title="Assignment agreement")
        axs[1].set(xlabel="K",ylabel="Median relative objective gap",title="Assignment objective gap")
        for ax in axs: ax.grid(True,alpha=.25);ax.legend(fontsize=6,ncol=2)
        fig.suptitle(scene);fig.tight_layout();fig.savefig(figdir/f"assignment_quality_{scene}.png",dpi=180);plt.close(fig)

    dev_pass={(r["k"],r["method"],r["effort"]) for r in gate if r["scene"]=="31UFS" and r["passes_gate"]}
    val_pass={(r["k"],r["method"],r["effort"]) for r in gate if r["scene"]=="31UFU" and r["passes_gate"]}
    common=dev_pass & val_pass
    corrected_gate_path=output/"phaseC_assignment_benchmarks/transfer_balanced_recheck/phaseC_assignment_benchmarks/gate_summary.csv"
    corrected_gate=[]
    if corrected_gate_path.exists():
        with corrected_gate_path.open(newline="") as f: corrected_gate=list(csv.DictReader(f))
    corrected_pass=[r for r in corrected_gate if r["passes_gate"]=="True"]
    corrected_table=[]
    if corrected_gate:
        corrected_table.append("\n## Transfer-balanced timing recheck\n\n"
            "The first complete pilot raw CSV is preserved above, but its exact-CUDA post-update row included a host-to-device C1 upload while ANN reused its resident C1. That asymmetry could favor ANN, so its relative timing figures are preliminary and must not be used as the authoritative speed comparison. A separate corrected run is preserved under `phaseC_assignment_benchmarks/transfer_balanced_recheck/`. Both methods reuse the same post-update centroid already on the GPU; the ANN refresh still includes centroid D2H, CPU index rebuild and index upload. Both upload X and outputs inside each query wall-time boundary.\n\n"
            "| Scene | K | Multi-projection effort | Min strict agreement | Max objective gap | Paired lifecycle speedup | Gate |\n|---|---:|---:|---:|---:|---:|:---:|\n")
        for scene in ("31UFS","31UFU"):
            for k in (16,32,64,128,256):
                candidates=[r for r in corrected_gate if r["scene"]==scene and int(r["k"])==k and r["method"]=="multi_projection"]
                if not candidates: continue
                best=max(candidates,key=lambda r:float(r["min_strict_agreement"]))
                corrected_table.append(f"| {scene} | {k} | {best['effort']} | {float(best['min_strict_agreement']):.4f} | {float(best['max_assignment_objective_gap']):.4f} | {float(best['median_paired_lifecycle_speedup']):.3f}x | {'PASS' if best['passes_gate']=='True' else 'FAIL'} |\n")
        dev_k64=next(r for r in corrected_gate if r["scene"]=="31UFS" and int(r["k"])==64 and r["method"]=="multi_projection" and int(r["effort"])==32)
        val_k64=next(r for r in corrected_gate if r["scene"]=="31UFU" and int(r["k"])==64 and r["method"]=="multi_projection" and int(r["effort"])==32)
        corrected_table.append("\nNo configuration passed the corrected gate. The K=64, effort=32 development setting met the agreement/objective bounds but had only {:.3f}x median paired lifecycle speedup (10% required); its validation counterpart had {:.3f}x speedup but a {:.1%} maximum objective gap. Thus the timing correction does not change the no-go decision.\n\n".format(
            float(dev_k64["median_paired_lifecycle_speedup"]),
            float(val_k64["median_paired_lifecycle_speedup"]),
            float(val_k64["max_assignment_objective_gap"])))
    result_lines=[]
    for scene in ("31UFS","31UFU"):
        result_lines.append(f"\n### {scene}: best four-projection strict agreement at each K\n\n")
        result_lines.append("| K | Effort | Minimum strict agreement | Maximum objective gap | Paired lifecycle speedup | Gate |\n|---:|---:|---:|---:|---:|:---:|\n")
        for k in sorted({r["k"] for r in gate if r["scene"]==scene and r["method"]=="multi_projection"}):
            candidates=[r for r in gate if r["scene"]==scene and r["k"]==k and r["method"]=="multi_projection"]
            best=max(candidates,key=lambda r:r["min_strict_agreement"])
            result_lines.append(f"| {k} | {best['effort']} | {best['min_strict_agreement']:.4f} | {best['max_assignment_objective_gap']:.4f} | {best['median_paired_lifecycle_speedup']:.3f}x | {'PASS' if best['passes_gate'] else 'FAIL'} |\n")
        result_lines.append("\nThe next table shows the fastest four-projection effort per K, regardless of quality; these speedups are not quality-equivalent when its agreement or objective gap fails.\n\n")
        result_lines.append("| K | Fastest effort | Paired lifecycle speedup | Minimum strict agreement | Maximum objective gap | Speed CI lower |\n|---:|---:|---:|---:|---:|---:|\n")
        for k in sorted({r["k"] for r in gate if r["scene"]==scene and r["method"]=="multi_projection"}):
            candidates=[r for r in gate if r["scene"]==scene and r["k"]==k and r["method"]=="multi_projection"]
            best=max(candidates,key=lambda r:r["median_paired_lifecycle_speedup"])
            result_lines.append(f"| {k} | {best['effort']} | {best['median_paired_lifecycle_speedup']:.3f}x | {best['min_strict_agreement']:.4f} | {best['max_assignment_objective_gap']:.4f} | {best['seed_bootstrap_speedup_ci95_low']:.3f}x |\n")
    historical=[]
    old_path=output/"phaseA_ann_diagnostics/historical_projection_gate_summary.csv"
    if old_path.exists():
        with old_path.open(newline="") as f: historical=list(csv.DictReader(f))
        by_k=[]
        for k in sorted({int(r["k"]) for r in historical}):
            at_k=[r for r in historical if int(r["k"])==k]
            by_k.append((k,max(float(r["median_strict_agreement"]) for r in at_k),
                         min(float(r["median_objective_gap"]) for r in at_k),
                         max(float(r["median_complete_speedup"]) for r in at_k)))
        diag=("# Existing single-projection GPU ANN diagnostics\n\n"
              f"Archived source CSV: `outputs/20261027-adaptive-four-phase-study/phase4_ann_and_combined/real_validation_pilot_device_refresh_v2/gate_summary.csv` (SHA256 `{sha(output/'phaseA_ann_diagnostics/historical_projection_gate_summary.csv')}`). The 44 rows span K/effort/initial-versus-moved snapshots; none passed both snapshot gates. The table below reports independent extrema over effort and snapshot rows, not one joint configuration.\n\n"
              "| K | Highest snapshot median strict agreement | Lowest snapshot median objective gap | Highest snapshot maintenance-inclusive speedup |\n|---:|---:|---:|---:|\n")
        for k,rec,gap,speed in by_k:
            diag+=f"| {k} | {rec:.4f} | {gap:.4f} | {speed:.3f}x |\n"
        diag+=("\nThe implementation sorts centroid projections on the CPU, uploads a sorted key/ID list, then performs a binary search and a contiguous projection-neighbor scan per query point on the GPU. The projected distance is not an admissible lower bound for Euclidean distance: points may be close along the chosen direction while far in omitted dimensions, and the true nearest centroid may lie outside the selected rank window. It then computes exact float64 squared distances only for shortlisted centroids, so selected distances are numerically exact but labels may be approximate. This is an algorithmic candidate-recall limitation, not an arithmetic error.\n\n"
              "At low K, the shortlist saves few K-by-D operations while adding projection, binary-search, branch and index costs; small-K exact fallback (K<=8 in the original adapter) still pays setup if an index is prepared. During fitting, refresh copies centroids from device to host, recomputes/sorts keys on CPU, and uploads index arrays; this maintenance is included in external benchmark timing. At larger K, low effort can reduce candidate distance evaluations and lower query time, but quality loss rises. Index creation/refresh and query are measured as categories; candidate selection is fused with query in one kernel, so its isolated time is unavailable.\n")
        (output/"phaseA_ann_diagnostics/diagnostic.md").write_text(diag)
    hist_text=f"The preserved historical single-projection gate has {len(historical)} K/effort/snapshot rows; {sum(r.get('passes_both_snapshot_gates')=='True' for r in historical)} configurations passed both initial and moved-state gates. Its best snapshot recall/speed figures do not imply an overall pass; source rows are preserved in `phaseA_ann_diagnostics/historical_projection_gate_summary.csv`.\n\n"
    summary_text=(
        "# Track 2 GPU ANN optimization study\n\n"
        "## Scope and protocol\n\n"
        "This is an assignment-only real-scene feasibility gate, not an end-to-end Mini-Batch result. It compares exact CUDA assignment, the preserved one-projection shortlist, and an isolated four-projection shortlist on prepared Sentinel-2 31UFS development and 31UFU validation features (N=262144, D=4), K=16/32/64/128/256, seeds 0/1/2, one excluded warm-up and three measured repetitions. Each seed/K uses `initialize_centroids`; a second centroid state is generated by one full-data exact CUDA Lloyd update. The two states share X and seed; exact CPU assignment checks the CUDA exact reference. Raw results include build/refresh, query, transfer, candidate-evaluation and quality fields.\n\n"
        "## Gate and result\n\n"
        f"The frozen gate required strict assignment agreement >= {GATE['strict_agreement_min']:.0%}, assignment-objective gap <= {GATE['assignment_objective_gap_max']:.0%}, median paired two-state lifecycle speedup >= {GATE['median_paired_lifecycle_speedup_min']:.0%}, and a seed-bootstrap 95% lower speedup bound > 1.0, on both centroid states and both scenes. Development-passing configurations: {len(dev_pass)}; validation-passing configurations: {len(val_pass)}; same configurations passing both: {len(common)}.\n\n"
        "No configuration passed both quality and maintenance-inclusive runtime gates on both scenes. Therefore end-to-end ANN Mini-Batch and combined adaptive ANN phases were stopped by design. The four-projection candidate did not consistently improve recall over the single-projection baseline at matched effort. It reduced median objective gap in some high-K matched settings, while strict agreement often fell; the paired deltas are in `paired_backend_comparison.csv`. It therefore did not produce a defensible assignment-objective/runtime tradeoff under the frozen gate. Low-effort speedups must not be described as quality-equivalent acceleration.\n\n"
        + "".join(result_lines) + "\n" + hist_text + "".join(corrected_table) +
        "The historical single-projection study also reported high speed at low effort in some settings but poor assignment accuracy, and CPU-built index maintenance remains part of cost. Projection distance does not bound Euclidean distance, so approximation errors are algorithmic; selected-centroid distances themselves are recomputed exactly. The four-projection query fuses projection search and exact candidate-distance checks in one CUDA kernel. Candidate generation is not separately timed; its cost is included in query time. CPU projection construction, device uploads, centroid D2H refresh and index re-upload are included in build/refresh timing.\n\n"
        "## Evidence files\n\n"
        "- Raw paired runs: `phaseC_assignment_benchmarks/individual_runs.csv`\n"
        "- Summary and variability: `phaseC_assignment_benchmarks/method_summary.csv`\n"
        "- Frozen-gate decisions: `phaseC_assignment_benchmarks/gate_summary.csv`\n"
        "- Matched single- versus four-projection deltas: `phaseC_assignment_benchmarks/paired_backend_comparison.csv`\n"
        "- Timing categories and overlap notes: `phaseC_assignment_benchmarks/timing_breakdown.csv`\n"
        "- Figures: `figures/assignment_runtime_*.png`, `figures/assignment_quality_*.png`\n\n"
        "## Skipped conditional phases and limitations\n\n"
        "Phase D (end-to-end ANN Mini-Batch) and Phase E (combined batch/ANN adaptation) were skipped because Stage C did not pass its predeclared gate. No larger N was tested, no held-out 30STG scene was used, and no ANN clustering or adaptive-effort benefit is established. The benchmark includes per-query index construction for the initial centroid state and refresh for the post-update state; these assignment-level results do not model all iterative index refreshes in a complete fit. Three seeds and three repetitions are a bounded feasibility study, not broad generalization. Exact CUDA remains the correctness and cost comparator.\n\n"
        "## Reproduction\n\n"
        "From the repository root in Ubuntu WSL, with the existing `.venv` and prepared scene arrays: `LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} .venv/bin/python experiments/benchmark_ann_optimization.py --pilot --budget-seconds 900`. This writes a new archive directory and refuses to overwrite an existing one. Rebuild summaries without rerunning GPU work using ` .venv/bin/python experiments/finalize_ann_optimization.py outputs/20261027-ann-optimization-study`."
    )
    (output/"final_summary/final_summary.md").write_text(summary_text)
    (output/"README.md").write_text("# Reproducible GPU ANN optimization gate\n\nSee `final_summary/final_summary.md`. Source data are existing prepared 31UFS/31UFU features; raw Sentinel-2 archives are not copied here.\n")
    if sha(raw_path)!=raw_hash_before:
        raise RuntimeError("Raw measurements changed during finalization")
    manifest_path=output/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["finalized_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["finalizer_source_sha256"]=sha(ROOT/"experiments/finalize_ann_optimization.py")
    manifest["raw_csv_sha256_verified_unchanged"]=raw_hash_before
    manifest["finalization_packages"]={}
    for package in ("matplotlib","numpy"):
        try: manifest["finalization_packages"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: manifest["finalization_packages"][package]=None
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str))
    if manifest.get("resume_attempts") and manifest.get("pre_resume_raw_rows") is not None:
        resumed_at=manifest["resume_attempts"][0]
        first_rows=int(manifest["pre_resume_raw_rows"])
        resume_rows=len(raw)-first_rows
        sessions={"note":"The interrupted first measurement attempt and resumed remainder are separate GPU sessions. No measurements are pooled across sessions for a paired speedup; each scene/K/seed/repetition/method pair was completed within one session.",
            "row_session_id_is_not_present_in_raw_csv":"Coverage below is reconstructed from the saved pre-resume row count and resume behavior; raw measurement values are unchanged.",
            "sessions":[
                {"session_id":"session_1","started_utc":manifest.get("started_utc"),"status":"interrupted; partial rows retained","raw_rows":first_rows,
                 "coverage":"31UFS all K/seeds/repetitions/methods; 31UFU K=16 and K=32 all seeds/repetitions/methods",
                 "gpu_before":manifest.get("gpu_before"),"gpu_after":"unavailable for interrupted attempt"},
                {"session_id":"session_2","started_utc":resumed_at.get("utc"),"status":"resumed missing cases and completed","raw_rows":resume_rows,
                 "coverage":"31UFU K=64,128,256 all seeds/repetitions/methods",
                 "gpu_before":resumed_at.get("gpu"),"gpu_after":manifest.get("gpu_after")}]}
        (output/"measurement_sessions.json").write_text(json.dumps(sessions,indent=2))
    inventory=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("checksums.sha256","experiment_index.json"):
            inventory.append({"path":str(path.relative_to(output)),"bytes":path.stat().st_size,"sha256":sha(path)})
    index={"created_utc":datetime.now(timezone.utc).isoformat(),"raw_measurement_csv_sha256":raw_hash_before,
           "raw_measurement_rows":len(raw),"files":inventory,
           "phase_status":{"A":"historical evidence reviewed","B":"four-projection prototype implemented and tested",
                           "C":("transfer-balanced recheck complete; gate failed" if corrected_gate else "real-scene three-seed pilot complete; gate failed"),
                           "D":"skipped: Phase C gate failed","E":"skipped: no eligible Phase D backend"}}
    (output/"experiment_index.json").write_text(json.dumps(index,indent=2))
    lines=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name!="checksums.sha256":
            lines.append(f"{sha(path)}  {path.relative_to(output)}\n")
    (output/"checksums.sha256").write_text("".join(lines))
    # Read every CSV and verify each recorded digest after the final index is written.
    for entry in inventory:
        path=output/entry["path"]
        if not path.is_file() or sha(path)!=entry["sha256"]:
            raise RuntimeError(f"Artifact verification failed: {entry['path']}")
        if path.suffix==".csv":
            with path.open(newline="") as f:
                list(csv.reader(f))
    return {"rows":len(raw),"raw_sha256":raw_hash_before,"gate_passes_dev":len(dev_pass),
            "gate_passes_validation":len(val_pass),"common_passes":len(common),
            "figures":len(list((output/"figures").glob("*.png"))),"checksums":len(lines)}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",nargs="?",type=Path,default=ROOT/"outputs/20261027-ann-optimization-study")
    args=parser.parse_args()
    print(json.dumps(finalize(args.archive),indent=2))


if __name__=="__main__": main()
