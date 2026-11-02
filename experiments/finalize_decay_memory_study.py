#!/usr/bin/env python3
"""Create derived tables, figures, report, index and hashes for the study."""

import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ARCHIVE=Path(os.environ.get("PCAP_DECAY_ARCHIVE", ROOT/"outputs/20261027-decay-memory-study")).resolve()
DEV=ARCHIVE/"phase1_development_retry1"
VAL=ARCHIVE/"phase2_validation"
DIAG=ARCHIVE/"phase0_convergence_diagnostic_retry1"
FINAL=ARCHIVE/"final_summary"
N=262144
from experiments.decay_memory_study import within_batch_oldest_to_newest_weight_ratio

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
    csv.field_size_limit(64*1024*1024)
    with p.open(newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def med(xs):return statistics.median([float(x) for x in xs if x not in (None,"")])
def iqr(xs):
    v=np.asarray([float(x) for x in xs if x not in (None,"")])
    return float(np.percentile(v,75)-np.percentile(v,25)) if len(v) else None

def main():
    FINAL.mkdir(exist_ok=True)
    val=read_csv(VAL/"individual_runs.csv")
    reconciled=[]
    for old in val:
        r=dict(old)
        r["raw_device_field"] = r.get("device")
        r["execution_device"] = "cuda"
        if r["method"]=="G0_EXACT_CUDA":
            steps=int(r["steps"])
            r["raw_observations_processed_field"] = r["observations_processed"]
            r["observations_processed"] = steps*N
            r["assignment_points_processed"]=(steps+1)*N
            r["work_count_definition"]="steps*N Lloyd update assignments; separate all-pass count includes initial and final exact assignment"
            if r.get("target_hit","").lower()=="true":
                r["time_to_target_s"]=r["wall_s"]
                r["target_time_scope"]="fit-completion upper bound; per-iteration wall timestamps unavailable"
        else:
            r["assignment_points_processed"]="not separately instrumented; includes update batches, probes, checkpoints and final assignment"
            r["work_count_definition"]="processed Mini-Batch observations; exact evaluation work is separately included in runtime"
        reconciled.append(r)
    write_csv(FINAL/"individual_runs_reconciled.csv",reconciled)

    methods=sorted({r["method"] for r in reconciled})
    timing_rows=read_csv(VAL/"timing_breakdown.csv")
    summary=[]
    for method in methods:
        rows=[r for r in reconciled if r["method"]==method]
        crossing=[r["first_target_crossing_time_s"] for r in rows if r.get("first_target_crossing_time_s") not in (None,"")]
        summary.append({"method":method,"n_fits":len(rows),"median_wall_s":med([r["wall_s"] for r in rows]),
                        "wall_iqr_s":iqr([r["wall_s"] for r in rows]),"median_final_exact_sse":med([r["final_exact_sse"] for r in rows]),
                        "final_sse_iqr":iqr([r["final_exact_sse"] for r in rows]),
                        "median_gap_to_exact_cuda_pct":100*med([r["relative_gap_to_exact_reference"] for r in rows]),
                        "target_hits":sum(r["target_hit"].lower()=="true" for r in rows),"target_hit_rate":sum(r["target_hit"].lower()=="true" for r in rows)/len(rows),
                        "target_hit_seeds":";".join(str(s) for s in range(5) if any(int(r["seed"])==s and r["target_hit"].lower()=="true" for r in rows)),
                        "median_observations":med([r["observations_processed"] for r in rows]),"median_steps":med([r["steps"] for r in rows]),
                        "median_first_target_crossing_observations":med([r["first_target_crossing_observations"] for r in rows if r.get("first_target_crossing_observations") not in (None,"")]) if any(r.get("first_target_crossing_observations") not in (None,"") for r in rows) else None,
                        "median_observed_target_crossing_s":med(crossing) if crossing else None,
                        "crossing_count":len(crossing),"termination_reasons":";".join(sorted(set(r["termination_reason"] for r in rows)))})
        trows=[r for r in timing_rows if r["method"]==method]
        for key in ("internal_sampling_s","internal_batch_assignment_s","internal_accumulation_update_s","internal_monitoring_s","internal_controller_s","internal_quality_evaluation_s","internal_final_assignment_s"):
            values=[r.get(key) for r in trows if r.get(key) not in (None,"")]
            summary[-1][key+"_median"]=med(values) if values else None
    write_csv(FINAL/"method_summary.csv",summary)
    write_csv(FINAL/"timing_summary.csv",[
        {"method":r["method"],"median_complete_fit_wall_s":r["median_wall_s"],
         "median_sampling_s":r["internal_sampling_s_median"],
         "median_batch_assignment_s":r["internal_batch_assignment_s_median"],
         "median_update_s":r["internal_accumulation_update_s_median"],
         "median_monitoring_s":r["internal_monitoring_s_median"],
         "median_controller_s":r["internal_controller_s_median"],
         "median_exact_quality_checkpoint_s":r["internal_quality_evaluation_s_median"],
         "median_final_assignment_s":r["internal_final_assignment_s_median"],
         "note":"Internal components are overlapping/subsets where specified; do not sum them to reconstruct wall time."}
        for r in summary])

    by={(r["method"],int(r["seed"]),int(r["repeat"])):r for r in reconciled}
    paired=[]
    for fixed in [m for m in methods if m.startswith("FIXED_DECAY")]:
        for seed in range(5):
            for rep in range(1,4):
                a=by[("ADAPTIVE_V2",seed,rep)];b=by[(fixed,seed,rep)]
                exact=float(a["relative_gap_to_exact_reference"])
                ratio=float(a["wall_s"])/float(b["wall_s"])
                a_cross=a.get("first_target_crossing_time_s") or ""
                b_cross=b.get("first_target_crossing_time_s") or ""
                paired.append({"adaptive_method":"ADAPTIVE_V2","fixed_method":fixed,"seed":seed,"repeat":rep,
                               "paired_v2_over_fixed_wall_ratio":ratio,"v2_wall_s":a["wall_s"],"fixed_wall_s":b["wall_s"],
                               "v2_final_sse":a["final_exact_sse"],"fixed_final_sse":b["final_exact_sse"],
                               "v2_minus_fixed_sse":float(a["final_exact_sse"])-float(b["final_exact_sse"]),
                               "v2_minus_fixed_sse_gap_percentage_points":100*(float(a["relative_gap_to_exact_reference"])-float(b["relative_gap_to_exact_reference"])),
                               "v2_target_hit":a["target_hit"],"fixed_target_hit":b["target_hit"],
                               "both_target_hit":a["target_hit"]=="True" and b["target_hit"]=="True",
                               "v2_first_crossing_s":a_cross,"fixed_first_crossing_s":b_cross})
    write_csv(FINAL/"paired_v2_vs_fixed.csv",paired)

    # Per-seed paired summaries treat seeds as the independent units and three
    # repeats as technical timing repetitions, rather than pseudo-replication.
    seed_pairs=[]
    for fixed in [m for m in methods if m.startswith("FIXED_DECAY")]:
        for seed in range(5):
            aa=[by[("ADAPTIVE_V2",seed,r)] for r in range(1,4)]
            bb=[by[(fixed,seed,r)] for r in range(1,4)]
            seed_pairs.append({"fixed_method":fixed,"seed":seed,
                "v2_median_wall_s":med([r["wall_s"] for r in aa]),"fixed_median_wall_s":med([r["wall_s"] for r in bb]),
                "v2_over_fixed_wall_ratio":med([a["wall_s"] for a in aa])/med([b["wall_s"] for b in bb]),
                "v2_median_sse":med([r["final_exact_sse"] for r in aa]),"fixed_median_sse":med([r["final_exact_sse"] for r in bb]),
                "v2_median_gap_pct":100*med([r["relative_gap_to_exact_reference"] for r in aa]),
                "fixed_median_gap_pct":100*med([r["relative_gap_to_exact_reference"] for r in bb]),
                "v2_target_hit_rate":sum(r["target_hit"]=="True" for r in aa)/3,
                "fixed_target_hit_rate":sum(r["target_hit"]=="True" for r in bb)/3})
    write_csv(FINAL/"paired_seed_summary.csv",seed_pairs)

    # Diagnostic-only matched cumulative trace.
    diag_runs=read_csv(DIAG/"individual_runs.csv")
    diag_cps=read_csv(DIAG/"paired_checkpoint_comparison.csv")
    diag_summary=[]
    for r in diag_runs:
        diag_summary.append({"method":r["method"],"seed":r["seed"],"final_exact_sse":r["final_exact_sse"],
                             "observation_count":r["observations_processed"],"steps":r["steps"],
                             "diagnostic_wall_not_benchmark":r["wall_s_diagnostic_only"],
                             "sampling_stream_matches":r["sampling_stream_matches"],
                             "effective_counts_or_counts":r["cluster_counts_or_effective_counts"]})
    write_csv(FINAL/"cumulative_vs_decay_diagnostic_summary.csv",diag_summary)

    events=read_csv(VAL/"controller_events.csv")
    v2events=[r for r in events if r.get("method")=="ADAPTIVE_V2"]
    decisions={}
    transitions=0
    for r in v2events:
        if r.get("controller_decision"):
            d=json.loads(r["controller_decision"]); decisions[d.get("reason","unknown")]=decisions.get(d.get("reason","unknown"),0)+1
        if r.get("batch_size") not in (None,"") and r.get("next_batch_size") not in (None,"") and r["batch_size"]!=r["next_batch_size"]:transitions+=1
    control={"controller_event_rows":len(v2events),"actual_batch_transitions":transitions,"decision_reasons":decisions,
             "largest_batch":65536,"target_input_to_controller":False}
    (FINAL/"controller_summary.json").write_text(json.dumps(control,indent=2,sort_keys=True)+"\n")

    # Figure set from saved data only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=[r["method"] for r in summary]
    fig,ax=plt.subplots(figsize=(9,5.6))
    for s in summary:
        ax.scatter(s["median_wall_s"],s["median_final_exact_sse"],s=65,label=s["method"])
    ax.set(xlabel="Complete-fit median wall time (s)",ylabel="Final exact full-data SSE",title="Validation runtime–quality trade-off")
    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FINAL/"validation_runtime_sse.png",dpi=180);plt.close(fig)
    cps=read_csv(VAL/"checkpoints.csv")
    checkpoint_series={}
    for r in cps:
        scheduled=json.loads(r.get("observation_budgets_reached") or "[]")
        points=scheduled or [int(r["points_processed"])]
        for point in points:
            checkpoint_series.setdefault((r["method"],int(point)),[]).append(
                (float(r["full_data_sse"]),float(r["elapsed_s"]),int(r["points_processed"])))
    fig,ax=plt.subplots(figsize=(9,5.6))
    for m in sorted({r["method"] for r in cps}):
        groups={point:values for (method,point),values in checkpoint_series.items() if method==m}
        xx=sorted(groups); yy=[np.median([v[0] for v in groups[x]]) for x in xx]
        ax.plot(np.array(xx)/1e6,yy,marker="o",label=m)
    ax.set(xlabel="Processed observations (millions)",ylabel="Exact full-data SSE",title="Validation SSE at shared observation checkpoints")
    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FINAL/"validation_sse_observations.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.6))
    for m in sorted({r["method"] for r in cps}):
        groups={point:values for (method,point),values in checkpoint_series.items() if method==m}
        xx=sorted(groups)
        ax.plot([np.median([v[1] for v in groups[x]]) for x in xx],
                [np.median([v[0] for v in groups[x]]) for x in xx],marker="o",label=m)
    ax.set(xlabel="Elapsed fit wall time at checkpoint (s)",ylabel="Exact full-data SSE",title="Validation SSE versus elapsed wall time")
    ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FINAL/"validation_sse_walltime.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.2))
    ax.bar([r["method"] for r in summary],[r["target_hit_rate"] for r in summary])
    ax.set(ylabel="Frozen target hit rate",title="Target attainment by method",ylim=(0,1.05))
    ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FINAL/"validation_target_hit_rate.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.6))
    for m in methods:
        vals=[float(r["first_target_crossing_time_s"]) for r in reconciled if r["method"]==m and r.get("first_target_crossing_time_s") not in (None,"")]
        if vals:ax.scatter([m]*len(vals),vals,label=m)
    ax.set(ylabel="First observed target time (s)",title="Time to first exact evaluation target crossing")
    ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(FINAL/"validation_time_to_target.png",dpi=180);plt.close(fig)
    dc=read_csv(DIAG/"checkpoints.csv")
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for method in ("CUMULATIVE","DECAYED_FIXED_8192"):
        rs=[r for r in dc if r.get("method")==method]
        for seed in (0,1,2):
            sr=[r for r in rs if int(r["seed"])==seed]
            sr.sort(key=lambda x:int(x["points_processed"]))
            field="full_exact_sse" if method=="CUMULATIVE" else "full_data_sse"
            if sr:axs[0].plot([int(r["points_processed"])/1e6 for r in sr],[float(r[field]) for r in sr],marker="o",label=f"{method} seed {seed}")
    axs[0].set(xlabel="Processed observations (millions)",ylabel="Exact full-data SSE",title="Matched update-rule diagnostic")
    cum=read_csv(DIAG/"paired_checkpoint_comparison.csv")
    for seed in (0,1,2):
        sr=[r for r in cum if int(r["seed"])==seed]
        axs[1].plot([int(r["points_processed"])/1e6 for r in sr],[float(r["cumulative_checkpoint_centroid_movement"]) for r in sr],marker="o",label=f"cumulative seed {seed}")
    axs[1].set(xlabel="Processed observations (millions)",ylabel="Max centroid displacement since prior checkpoint",title="Cumulative centroid movement")
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(FINAL/"cumulative_decay_diagnostic.png",dpi=180);plt.close(fig)

    # Build evidence-based report text.
    tab={r["method"]:r for r in summary}
    adaptive=tab["ADAPTIVE_V2"]; best_fixed=min((tab[m] for m in tab if m.startswith("FIXED_DECAY")),key=lambda r:r["median_wall_s"])
    best_fixed_method=best_fixed["method"]
    pair_best=[r for r in paired if r["fixed_method"]==best_fixed_method]
    pair_ratio=med([r["paired_v2_over_fixed_wall_ratio"] for r in pair_best])
    pair_gap=med([r["v2_minus_fixed_sse_gap_percentage_points"] for r in pair_best])
    pair_cross=[r for r in pair_best if r["v2_first_crossing_s"] not in (None,"") and r["fixed_first_crossing_s"] not in (None,"")]
    g0=tab["G0_EXACT_CUDA"]
    c_sse=[float(r["final_exact_sse"]) for r in diag_runs if r["method"]=="CUMULATIVE"]
    d_sse=[float(r["final_exact_sse"]) for r in diag_runs if r["method"]=="DECAYED_FIXED_8192"]
    sse_delta=[d-c for d,c in zip(d_sse,c_sse)]
    pilot=read_csv(DEV/"individual_runs.csv")
    pilot_summary={}
    for m in sorted({r["method"] for r in pilot}):
        rr=[r for r in pilot if r["method"]==m]
        pilot_summary[m]=(round(med([r["wall_s"] for r in rr]),3),round(med([r["final_exact_sse"] for r in rr]),2),sum(r["target_hit"]=="True" for r in rr))
    (FINAL/"summary.md").write_text(f"""# Decayed Mini-Batch convergence and Adaptive V2 study

## Scope and provenance

This study used prepared Sentinel-2 Level-2A features only: development scene 31UFS and validation scene 31UFU, each with N=262,144, D=4 and K=16. The validation matrix is {json.loads((VAL/'manifest.json').read_text())['inputs']['X_array_sha256']}; its per-seed C0 hashes were checked against the frozen-reference records. Methods used exact CUDA assignment, float64 accumulation, and final exact full-data SSE. Mini-Batch arms processed 40,960,000 observations and were evaluated at five shared full-data checkpoints. The frozen 5% target was used only by the analysis process, never passed to a fitting/controller function.

All runs used the physical RTX 4050 (CUDA simulator disabled). Complete fit wall times include preparation, GPU transfers, updates, scheduled full-data checks and exact final evaluation. Internal timing components are diagnostic subsets and must not be summed independently.

## Decay fairness

The unchanged block update is `S_t = lambda_b S_(t-1) + batch_sum`, `W_t = lambda_b W_(t-1) + batch_count`, `C_t=S_t/W_t`, with `lambda_b=0.5**(b/(10N))`. For a batch schedule with total processed observations M, retained prior-history mass is exactly `product(lambda_b)=0.5**(M/(10N))`; changing batch size does not change the historical-statistic half-life (2,621,440 observations for these runs). New observations within a batch receive equal weight, so the existing method is blockwise rather than exact per-observation exponential weighting. At batch 65,536 the omitted oldest/newest within-batch weight ratio would be {within_batch_oldest_to_newest_weight_ratio(65536,N):.5f}; we preserved the original rule and did not silently normalize it.

## Development pilot

Three seed runs per arm on 31UFS were used only to verify the protocol. All Mini-Batch arms processed 40.96M observations and none met the pilot's descriptive 5% exact-G0 criterion. Median (wall seconds, final SSE, target hits / 3): {pilot_summary}. The validation settings, methods, target and decision criteria were then frozen in `frozen_validation_protocol.json`; no held-out scene was used.

## Validation results

| Method | Median wall (s) [IQR] | Median exact SSE [IQR] | Gap to exact CUDA (%) | Target hits | Median updates | Median first crossing observations |
|---|---:|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {r['method']} | {r['median_wall_s']:.3f} [{r['wall_iqr_s']:.3f}] | {r['median_final_exact_sse']:.2f} [{r['final_sse_iqr']:.2f}] | {r['median_gap_to_exact_cuda_pct']:.3f} | {r['target_hits']}/{r['n_fits']} | {r['median_steps']:.0f} | {int(r['median_first_target_crossing_observations']) if r['median_first_target_crossing_observations'] is not None else 'not time-stamped'} |" for r in summary)+f"""

Exact CUDA G0 converged in a median {g0['median_wall_s']:.3f} s and met the frozen target in {g0['target_hits']}/{g0['n_fits']} fits. Its first target-crossing assignment was observed at a median {int(g0['median_first_target_crossing_observations']):,} update-point presentations (26 full Lloyd updates); those are not equivalent to Mini-Batch observations, and its crossing time was not timestamped. Its complete-fit time is an upper bound. Every Mini-Batch method used the full observation budget and terminated by `observation_budget`, not convergence. Mini-Batch target hits occurred for seeds 0 and 2 only (6/15 repetitions per method); seeds 1, 3 and 4 missed for all Mini-Batch variants. Repetitions reuse each seed's deterministic sampled-index stream; GPU floating-point atomics can still create small run-to-run numerical variation, so the independent units for seed variability are five seeds, not 15 independent data draws.

Median monitoring/controller internal times are preserved in `timing_summary.csv`; as documented in each run's timing scope, these component values are diagnostic subsets and must not be added to complete-fit time.

The fastest fixed arm by median complete-fit wall time was {best_fixed_method} ({best_fixed['median_wall_s']:.3f} s). Paired Adaptive V2 / {best_fixed_method} median runtime ratio was {pair_ratio:.3f} (ratio >1 means V2 was slower); median paired final-SSE gap difference was {pair_gap:.3f} percentage points relative to exact CUDA. Adaptive V2 had lower final SSE in {sum(float(r['v2_minus_fixed_sse'])<0 for r in pair_best)}/{len(pair_best)} paired runs, but it did not meet the predeclared 0.5 percentage-point equivalence band and was slower. It therefore did not establish an adaptive efficiency advantage over the best fixed batch. {len(pair_cross)} paired runs had an observed target crossing in both methods; their crossings are checkpoint-censored. Exact G0 first-crossing wall time is unavailable; its complete fit time is reported as an upper bound.

## Cumulative versus decayed diagnostic

A separate trace-only validation diagnostic compared cumulative fixed-batch and fixed-decay batch-8192 runs for seeds 0–2, same X, same C0, same 40.96M observations, same 8.192M checkpoints, same batch size, and identical sampled-index hash stream for every update. It added host diagnostic copies and is **not** a runtime benchmark. Final exact SSE (cumulative → decayed) was {[(round(c,2),round(d,2)) for c,d in zip(c_sse,d_sse)]}; decayed minus cumulative SSE was { [round(x,2) for x in sse_delta] }. Decayed had lower SSE in all three matched seeds. Cumulative SSE continued to decline at every checkpoint; its max centroid displacement between checkpoints contracted across the run but remained nonzero. This supports cumulative-history inertia as a contributor in these seeds, while not showing that cumulative Mini-Batch had reached a plateau. The three-seed diagnostic is narrow evidence, not a general result.

## Controller and schedule interpretation

Adaptive V2's 15 validation fits made {transitions} actual batch-size changes. The observed growth pattern reached 65,536; decisions were driven by fixed-probe objective stagnation/progress signals only. Repeated no-op decisions at the maximum were logged, not counted as changes. V2 can improve SSE and reduce time relative to smaller fixed batches, but the largest fixed batch was faster. The conditional schedule replay was skipped because the predeclared advantage gate failed: V2 did not beat the fastest fixed arm at the required comparable quality and did not improve time-to-target.

## Answers and limits

1. Cumulative Mini-Batch missed the target because its SSE was still improving but slowly at the cap; same-work, same-sample matched diagnostics show decayed updates yielded lower SSE in all three tested seeds. This makes cumulative-count inertia plausible, but it is not the only possible contributor and the validation sample is small.
2. Decay explains a substantial part of the observed SSE difference in the matched 3-seed diagnostic; this is evidence for the tested setup, not a universal guarantee.
3. Adaptive V2 did not beat the fastest fixed-decay method at comparable quality under the frozen criterion.
4. V2 reached target at checkpoints for the same two seed groups as fixed arms. The fastest fixed batch crossed earlier in the observed crossings; V2 did not improve time-to-target.
5. V2's benefit versus small fixed batches is not evidence of feedback advantage over the best fixed batch. The optional schedule replay was not justified by the predeclared gate.
6. The strongest defensible result is that observation-scaled decay improved fixed-batch convergence quality versus cumulative updates in a tightly matched three-seed validation diagnostic, while Adaptive V2 improved the quality/runtime trade-off versus smaller fixed batches but did not improve efficiency over batch 65,536 or exact CUDA G0.

Limitations: one 512×512 validation crop and one development crop; five validation seeds with three repeated fits per method (repetitions reuse the same sample stream); no ground-truth land-cover labels; fixed 4-feature input; Mini-Batch time-to-target sampled only at 8.192M-observation exact evaluation checkpoints; exact CUDA G0 has no per-iteration wall timestamps; trace-only cumulative timing is instrumented and not comparable; no schedule replay; observed CUDA occupancy warnings; no causal generalization beyond these data/configurations.

## Artifacts

- Raw fit rows: `phase2_validation/individual_runs.csv` (preserved; its original G0 observation-count field is 0 because that implementation uses a different metadata key).
- Corrected derived rows: `final_summary/individual_runs_reconciled.csv` (G0 update-observation counts derived as `steps*N`; includes all-pass assignment count).
- Checkpoints, controller history, timing components and figures are adjacent in the phase directories and `final_summary/`.
- Matched cumulative diagnostic: `phase0_convergence_diagnostic_retry1/`.
- Preflight-only failures are retained in `phase1_development/startup_failure.json` and `phase0_convergence_diagnostic/startup_failure.json`; no fit was executed in those attempts.
- Exact configuration freeze: `frozen_validation_protocol.json`.
""")
    def cmd_version(command):
        try:return subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip()
        except Exception as exc:return f"unavailable: {exc}"
    original_manifest=json.loads((VAL/"manifest.json").read_text())
    provenance={
        "validation_run_source_hashes":original_manifest.get("source_hashes",{}),
        "current_source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/"experiments/run_decay_memory_study.py",ROOT/"experiments/decay_memory_study.py",
            ROOT/"experiments/run_cumulative_decay_diagnostic.py",ROOT/"experiments/finalize_decay_memory_study.py",
            ROOT/"src/pcap/budgeted_minibatch.py",ROOT/"src/pcap/minibatch_decayed_cuda.py",
            ROOT/"src/pcap/adaptive_v2_controller.py",ROOT/"src/pcap/kmeans_cuda.py"]},
        "post_run_source_correction":{"file":"experiments/run_decay_memory_study.py",
            "reason":"exact CUDA fit metadata does not expose the Mini-Batch points_processed key; the original raw run CSV records zero in that field. The reconciled copy derives Lloyd update assignments as steps*N and all exact assignment passes as (steps+1)*N. No fits or raw rows were changed."},
        "post_run_instrumentation_cli_change":{"file":"experiments/run_cumulative_decay_diagnostic.py",
            "reason":"an --output-dir option was added after the diagnostic solely to make future reproductions write to a fresh archive path; no update, sampling, metric, or saved diagnostic result changed."},
        "nvcc_version":cmd_version(["nvcc","--version"]),
        "package_versions":json.loads((VAL/"manifest.json").read_text()).get("environment",{}).get("packages",{}),
    }
    (FINAL/"source_and_measurement_corrections.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    (ARCHIVE/"README.md").write_text(f"""# Reproducing the decayed Mini-Batch study

This archive contains the 31UFS development pilot, frozen protocol, 31UFU validation study and the separate trace-only cumulative diagnostic. Raw experiment outputs are preserved. The held-out scene was not used. No schedule replay was run because the predeclared gate did not pass.

## Environment and GPU

Use the repository's Python 3.12 `.venv` and installed lock versions listed in each stage manifest. On WSL2, expose the WSL driver libraries for the command only:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${{LD_LIBRARY_PATH:-}}"
nvidia-smi
```

The runner rejects the CUDA simulator and refuses CPU fallback. The archived run used a real RTX 4050 Laptop GPU, driver 581.86, compute capability 8.9. See the stage manifests for clocks, temperature, utilization, RAM and package versions.

## Reproduce in a new archive

Do not rerun into this preserved output directory. Select a fresh ignored output root, copy the frozen protocol file unchanged, then execute:

```bash
export PCAP_DECAY_ARCHIVE="outputs/20261101-decay-memory-study-reproduction"
mkdir -p "$PCAP_DECAY_ARCHIVE"
cp outputs/20261027-decay-memory-study/frozen_validation_protocol.json "$PCAP_DECAY_ARCHIVE/"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${{LD_LIBRARY_PATH:-}}"
.venv/bin/python experiments/run_decay_memory_study.py --stage phase1_development --archive-root "$PCAP_DECAY_ARCHIVE"
.venv/bin/python experiments/run_decay_memory_study.py --stage phase2_validation --archive-root "$PCAP_DECAY_ARCHIVE"
.venv/bin/python experiments/run_cumulative_decay_diagnostic.py --output-dir "$PCAP_DECAY_ARCHIVE/phase0_convergence_diagnostic"
PCAP_DECAY_ARCHIVE="$PCAP_DECAY_ARCHIVE" .venv/bin/python experiments/finalize_decay_memory_study.py
```

The frozen validation targets remain the exact per-seed targets in the existing `full_disjoint_probe/reference_runs.csv` and are never arguments to algorithm/controller calls.

## Archive contents and interpretation

- `phase1_development_retry1/`: development fits, warm-ups, exact references, checkpoints and schedules. The earlier `phase1_development/` directory contains a pre-fit runner failure and is retained.
- `frozen_validation_protocol.json`: settings and success criteria committed before validation.
- `phase2_validation/`: 90 individual fits, per-update sample-hash/controller history, exact checkpoints, timing components, config, manifest and hashes.
- `phase0_convergence_diagnostic_retry1/`: matched three-seed cumulative/decayed tracing; diagnostic timings include trace overhead and are not performance measurements. The first pre-fit failure remains in `phase0_convergence_diagnostic/`.
- `final_summary/summary.md`: complete measured interpretation, negative findings, methodology limits and numerical tables.
- `final_summary/individual_runs_reconciled.csv`: derived correction for exact CUDA work-count metadata; the raw validation CSV is unchanged.
- `checksums.sha256` and `experiment_index.json`: checksums and machine-readable archive inventory.

The five validation seeds are the independent data/initialization units. The three repetitions reuse each seed's deterministic sampling stream and primarily characterize runtime variation; do not treat all 15 runs as independent seeds. Exact CUDA assignment-point work and Mini-Batch observations are different work units, not interchangeable counts.
""")
    # Root-level index/checksums. Historical source archives are included as
    # read-only references by path, not copied into this new study.
    entries=[]
    for p in sorted(ARCHIVE.rglob("*")):
        if p.is_file() and p.name not in ("checksums.sha256","experiment_index.json"):
            entries.append({"path":str(p.relative_to(ARCHIVE)),"sha256":sha(p),"bytes":p.stat().st_size})
    index={"archive":"outputs/20261027-decay-memory-study","completed_stages":["phase1_development_retry1","phase2_validation","phase0_convergence_diagnostic_retry1"],"failed_preflight_attempts":["phase1_development","phase0_convergence_diagnostic"],"source_provenance":"final_summary/source_and_measurement_corrections.json","files":entries}
    (ARCHIVE/"experiment_index.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
    checksum=[]
    for p in sorted(ARCHIVE.rglob("*")):
        if p.is_file() and p.name!="checksums.sha256":checksum.append(f"{sha(p)}  {p.relative_to(ARCHIVE)}")
    (ARCHIVE/"checksums.sha256").write_text("\n".join(checksum)+"\n")
    print(f"wrote derived summary for {len(reconciled)} validation fits; files indexed: {len(entries)}")

if __name__=="__main__":main()
