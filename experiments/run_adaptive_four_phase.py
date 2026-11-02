"""Run bounded, observation-budget-matched adaptive CUDA experiments.

This runner preserves production baselines and records the experimental
disjoint-probe protocol. Each output directory must be new.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.kmeans_cpu import fit_cpu
from pcap.controlled_fit import fit_controlled
from pcap.adaptive_controller import BatchPolicy
from pcap.quality_guard_controller import QualityGuardPolicy
from pcap.budgeted_minibatch import fit_budgeted_minibatch
from run_suite import provenance, snapshot_sources, array_hash, validate_result


def _write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict,list,tuple)) else v
                        for k,v in row.items()})


def _sha(data): return hashlib.sha256(data).hexdigest()


def _summarize(output, runs, checkpoints):
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in runs: grouped[r["method"]].append(r)
    summary=[]
    for method, rows in sorted(grouped.items()):
        walls=np.asarray([float(x["wall_s"]) for x in rows])
        sses=np.asarray([float(x["final_sse"]) for x in rows])
        summary.append(dict(method=method,execution_device=rows[0]["device"],runs=len(rows),
            median_wall_s=float(np.median(walls)),wall_iqr_s=float(np.percentile(walls,75)-np.percentile(walls,25)),
            median_final_exact_sse=float(np.median(sses)),target_hits=int(sum(bool(x["target_hit"]) for x in rows)),
            target_hit_rate=float(np.mean([bool(x["target_hit"]) for x in rows])),
            median_observations=float(np.median([float(x["observations_processed"]) for x in rows])),
            termination_reasons=";".join(sorted(set(str(x["termination_reason"]) for x in rows)))))
    _write_csv(output/"method_summary.csv",summary)
    if any(r.get("phase")=="phase2" for r in runs):
        pairs=[]
        for cpu,gpu in (("C1","G1"),("C3","G3CUDA")):
            by_cpu={(r["seed"],r["repeat"]):r for r in runs if r["method"]==cpu}
            by_gpu={(r["seed"],r["repeat"]):r for r in runs if r["method"]==gpu}
            for key in sorted(set(by_cpu)&set(by_gpu)):
                a,b=by_cpu[key],by_gpu[key]
                valid=(a["X_sha256"]==b["X_sha256"] and a["C0_sha256"]==b["C0_sha256"] and
                       a["observations_processed"]==b["observations_processed"])
                same_policy=(cpu!="C3" or a["batch_schedule"]==b["batch_schedule"])
                pairs.append(dict(cpu_method=cpu,gpu_method=gpu,seed=key[0],repeat=key[1],
                    same_inputs=valid,same_observation_budget=valid,
                    batch_schedule_equal=same_policy,wall_ratio_cpu_over_gpu=float(a["wall_s"])/float(b["wall_s"]),
                    quality_matched=bool(a["target_hit"] and b["target_hit"]),
                    cpu_final_sse=a["final_sse"],gpu_final_sse=b["final_sse"],
                    cpu_target_hit=a["target_hit"],gpu_target_hit=b["target_hit"],
                    interpretation="paired complete-fit ratio; call matched-quality speedup only when both hit frozen target"))
        _write_csv(output/"paired_speedups.csv",pairs)
    # Internal categories are exported separately. Their sums are not a wall-time estimate.
    timing=[]
    for r in runs:
        t=r["timings"]
        timing.append(dict(method=r["method"],seed=r["seed"],repeat=r["repeat"],wall_s=r["wall_s"],
            setup_s=t.get("setup_s"),sampling_s=t.get("sampling_s"),batch_assignment_s=t.get("batch_assignment_s"),
            accumulation_update_s=t.get("accumulation_update_s"),monitoring_s=t.get("monitoring_s"),
            controller_s=t.get("controller_s"),quality_evaluation_s=t.get("quality_evaluation_s"),
            final_assignment_s=t.get("final_assignment_s"),components_overlap="see manifest; do not sum"))
    _write_csv(output/"timing_breakdown.csv",timing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output.mkdir(exist_ok=True)
        methods=sorted(grouped)
        fig,ax=plt.subplots(1,2,figsize=(11,4.2))
        ax[0].boxplot([[float(x["wall_s"]) for x in grouped[m]] for m in methods],tick_labels=methods)
        ax[0].set_ylabel("Complete fit wall time (s)");ax[0].tick_params(axis="x",rotation=25)
        ax[1].boxplot([[float(x["final_sse"]) for x in grouped[m]] for m in methods],tick_labels=methods)
        ax[1].set_ylabel("Final exact full-data SSE");ax[1].tick_params(axis="x",rotation=25)
        fig.tight_layout();fig.savefig(output/"runtime_and_sse.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in sorted(set(x["method"] for x in checkpoints)):
            subset=[x for x in checkpoints if x["method"]==method]
            by=defaultdict(list)
            for x in subset: by[int(x["points_processed"])].append(float(x["full_exact_sse"]))
            xx=sorted(by); yy=[float(np.median(by[i])) for i in xx]
            ax.plot(xx,yy,marker="o",label=method)
        ax.set_xlabel("Processed observations");ax.set_ylabel("Exact full-data SSE");ax.legend();fig.tight_layout()
        fig.savefig(output/"sse_vs_observations.png",dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in methods:
            subset=[x for x in checkpoints if x["method"]==method]
            by=defaultdict(list)
            for x in subset: by[int(x["points_processed"])].append(float(x["full_exact_sse"]))
            xx=sorted(by); yy=[float(np.median(by[i])) for i in xx]
            ax.plot(xx,yy,marker="o",label=method)
        ax.axhline(float(np.median([r["target_sse"] for r in runs])),linestyle="--",color="black",label="median evaluation target")
        ax.set_xlabel("Processed observations");ax.set_ylabel("Exact full-data SSE");ax.legend();fig.tight_layout()
        fig.savefig(output/"target_and_sse.png",dpi=180);plt.close(fig)
    except Exception as exc:
        (output/"plot_failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")


def _first_target_crossing(method, history, checkpoints, target_sse, n_points):
    """Return processed observations and elapsed seconds at the first exact hit."""
    if method == "G0":
        for item in history:
            if float(item.get("full_data_sse", float("inf"))) <= target_sse:
                observations = int(item.get("points_processed", int(item.get("step", 0)) * n_points))
                return observations, float(item["elapsed_s"])
    else:
        for item in checkpoints:
            if float(item["full_exact_sse"]) <= target_sse:
                return int(item["points_processed"]), float(item["elapsed_s"])
    return None, None


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    X = np.load(args.scene_dir / "X.npy", allow_pickle=False)
    scene = json.loads((args.scene_dir / "metadata.json").read_text())
    if X.dtype != np.float32 or not X.flags.c_contiguous or not np.isfinite(X).all():
        raise ValueError("prepared X must be finite contiguous float32")
    x_hash = _sha(X.tobytes())
    if scene.get("X_sha256") != x_hash: raise ValueError("prepared scene hash mismatch")
    if (X.shape[0],X.shape[1]) != (262144,4):
        raise ValueError(f"expected approved validation crop N=262144,D=4; found {X.shape}")
    phases = {
      "phase1": ["G0","G1","G3CUDA"],
      "phase2": ["G0","C1","G1","C3","G3CUDA"],
      "phase3": ["G0","G1","G3CUDA","QGCUDA"],
    }
    methods = phases[args.phase]
    policy = BatchPolicy(minimum=4096, maximum=65536, initial_batch=8192,
                         deterioration=.01, quiet_movement=.001,
                         quiet_label_change=.005, quiet_objective_change=.001,
                         oscillation_change=.001, quiet_checkpoints=10)
    qpolicy = QualityGuardPolicy(minimum_batch=4096, maximum_batch=65536,
                         initial_batch=8192, deterioration=.01,
                         stagnation_improvement=.001, progress_improvement=.005,
                         movement_limit=.001, label_change_limit=.005,
                         oscillation_change=.001, required_checkpoints=3,
                         dwell_checkpoints=2, adjustment_factor=2)
    config = dict(phase=args.phase, stage=args.stage, scene=scene,
        n=int(X.shape[0]), d=int(X.shape[1]), k=args.k,
        seeds=args.seeds, measured_repetitions=args.repeats, warmups=args.warmups,
        observation_budget=args.observation_budget,
        evaluation_interval=args.evaluation_interval, initial_batch=8192,
        probe_interval=10, tolerance=1e-6, quality_gap=.05,
        methods=methods, batch_policy=policy.__dict__, quality_guard_policy=qpolicy.__dict__,
        sampling=("uniform without replacement per update over complement of common fixed probe"
                  if args.phase=="phase3" else "production uniform sampling over all X; common fixed probe may overlap"),
        probe="seeded fixed 1024-row stream; disjoint in phase3, diagnostic/overlapping for phases1-2",
        disjoint_probe=(args.phase=="phase3"),
        controller_visibility="probe only; frozen reference and target stay in evaluator",
        timing="external perf_counter around full fit; target quality checks included; 2 warmups excluded",
        git_commit=provenance()["git_commit"], x_sha256=x_hash,
        cuda_simulator=False)
    (output/"config.json").write_text(json.dumps(config,indent=2,default=str))
    manifest = dict(provenance(), created_utc=datetime.now(timezone.utc).isoformat(),
        complete=False, failures=[], config=config, source_hashes=provenance()["source_hashes"],
        protocol_note="G1/G3CUDA update equations and exact assignment reused through a budgeted orchestration; common 1024-point probe is withheld from sampling. Fixed/adaptive fits stop at identical processed-observation checkpoints. This is a controlled replay, not a replacement for the production fit entry points.")
    snapshot_sources(output/"source",manifest["source_hashes"])
    warmup_rows=[]; reference_rows=[]; individual=[]; checkpoint_rows=[]; event_rows=[]
    # warmups are short and are recorded separately from measured observations.
    seed0=args.seeds[0]
    C0=initialize_centroids(X,args.k,seed=seed0)
    warm_budget=min(2*8192,args.observation_budget)
    for warm_idx in range(args.warmups):
        for method in methods:
            cfg=FitConfig(args.k,seed0,4,1e-6,None if method=="G0" else 8192,10)
            t=perf_counter()
            if method=="G0":
                fit_controlled(X,C0,cfg,device="cuda",mode="full",evaluation_interval=None)
            else:
                dev="cpu" if method in ("C1","C3") else "cuda"
                fit_budgeted_minibatch(X,C0,cfg,device=dev,observation_budget=warm_budget,
                    adaptive=method in ("G3CUDA","C3"),batch_policy=policy if method in ("G3CUDA","C3") else None,
                    quality_guard_policy=qpolicy if method=="QGCUDA" else None,
                    evaluation_interval=warm_budget,disjoint_probe=(args.phase=="phase3"))
            warmup_rows.append(dict(method=method,warmup=warm_idx,wall_s=perf_counter()-t,excluded=True))
    _write_csv(output/"warmups.csv",warmup_rows)
    if args.phase == "phase2":
        order_rng=np.random.default_rng(121)
    else:
        order_rng=np.random.default_rng(122)
    started=perf_counter()
    try:
        for seed in args.seeds:
            C0=initialize_centroids(X,args.k,seed=seed); c_hash=_sha(C0.tobytes())
            ref=fit_cpu(X,C0,FitConfig(args.k,seed,1000,1e-6))
            if ref.metadata.get("termination_reason")!="tolerance":
                raise RuntimeError(f"exact CPU reference for seed {seed} did not converge")
            target=ref.sse*1.05
            reference_rows.append(dict(seed=seed,C0_sha256=c_hash,reference_sse=ref.sse,
                reference_steps=ref.metadata["steps"],target_sse=target))
            for repeat in range(args.repeats):
                order=[methods[i] for i in order_rng.permutation(len(methods))]
                for method in order:
                    if perf_counter()-started>args.time_budget_seconds:
                        raise TimeoutError("phase execution budget exceeded between complete fits")
                    cfg=FitConfig(args.k,seed,args.max_steps,1e-6,
                        None if method=="G0" else 8192,10)
                    t=perf_counter()
                    if method=="G0":
                        result=fit_controlled(X,C0,cfg,device="cuda",mode="full",
                            evaluation_interval=1,target_sse=target)
                        points=result.metadata.get("points_processed",0)
                        cps=[dict(points_processed=int(entry["step"]*len(X)),
                                  full_exact_sse=entry["full_data_sse"],step=entry["step"],
                                  checkpoint_kind="lloyd_iteration")
                             for entry in result.metadata.get("quality_history",[])]
                        ev=[]
                    else:
                        dev="cpu" if method in ("C1","C3") else "cuda"
                        result,cps,ev=fit_budgeted_minibatch(X,C0,cfg,device=dev,
                            observation_budget=args.observation_budget,adaptive=method in ("G3CUDA","C3"),
                            batch_policy=policy if method in ("G3CUDA","C3") else None,
                            quality_guard_policy=qpolicy if method=="QGCUDA" else None,
                            evaluation_interval=args.evaluation_interval,
                            disjoint_probe=(args.phase=="phase3"))
                        points=result.metadata["points_processed"]
                        target_hit=any(x["full_exact_sse"]<=target for x in cps)
                        result.metadata["target_satisfied"] = bool(target_hit)
                        result.metadata["target_sse_evaluation_only"] = target
                        result.metadata["first_target_crossing_observations"] = next(
                            (x["points_processed"] for x in cps if x["full_exact_sse"]<=target),None)
                    elapsed=perf_counter()-t
                    validate_result(X,result)
                    hit=(result.sse<=target) if method=="G0" else result.metadata["target_satisfied"]
                    first_obs, first_time = _first_target_crossing(
                        method, result.metadata.get("quality_history", []), cps, target, len(X))
                    row=dict(phase=args.phase,method=method,device=result.metadata.get("execution_device", "cuda"),
                        seed=seed,repeat=repeat,n=len(X),d=X.shape[1],k=args.k,
                        X_sha256=x_hash,C0_sha256=c_hash,target_sse=target,final_sse=result.sse,
                        gap_to_reference=(result.sse-ref.sse)/ref.sse,target_hit=bool(hit),
                        target_crossing_observation=first_obs,target_crossing_time_s=first_time,
                        wall_s=elapsed,observations_processed=points,steps=result.metadata.get("steps"),
                        termination_reason=result.metadata.get("termination_reason"),
                        convergence_satisfied=result.metadata.get("convergence_satisfied",False),
                        committed_observations=result.metadata.get("committed_observations",points),
                        rolled_back_observations=result.metadata.get("observations_rolled_back",0),
                        timings=result.timings,metadata=result.metadata,
                        batch_schedule=[(h["points_processed"],h.get("requested_batch_size"),h.get("actual_batch_size"),h.get("next_batch_size")) for h in result.history])
                    individual.append(row)
                    for cp in cps:
                        checkpoint_rows.append(dict(method=method,seed=seed,repeat=repeat,**cp,
                            elapsed_time_scope="coarse fit wall only; exact checkpoint time not instrumented"))
                    for h in result.history:
                        if h.get("controller_decision") is not None:
                            event_rows.append(dict(method=method,seed=seed,repeat=repeat,**h))
                    for event in ev: event_rows.append(dict(method=method,seed=seed,repeat=repeat,**event))
                    _write_csv(output/"individual_runs.csv",individual)
                    _write_csv(output/"checkpoints.csv",checkpoint_rows)
                    _write_csv(output/"controller_events.csv",event_rows)
                    _write_csv(output/"reference_runs.csv",reference_rows)
                    _write_csv(output/"timing_breakdown.csv",[
                        dict(method=r["method"],seed=r["seed"],repeat=r["repeat"],**r["timings"])
                        for r in individual])
        _summarize(output,individual,checkpoint_rows)
        manifest["complete"]=True
    except Exception as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_wall_s"]=perf_counter()-started
        manifest["completed_measured_fits"]=len(individual)
        manifest["references"]=reference_rows
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
        files=[p for p in output.rglob("*") if p.is_file() and p.name!="checksums.sha256"]
        (output/"checksums.sha256").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n" for p in sorted(files)))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("phase1","phase2","phase3"),required=True)
    p.add_argument("--stage",choices=("pilot","full"),required=True)
    p.add_argument("--scene-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--warmups",type=int,default=2)
    p.add_argument("--observation-budget",type=int,default=8192000)
    p.add_argument("--evaluation-interval",type=int,default=2048000)
    p.add_argument("--max-steps",type=int,default=10000)
    p.add_argument("--time-budget-seconds",type=float,default=900)
    p.add_argument("--k",type=int,default=16)
    a=p.parse_args(argv)
    a.scene_dir=a.scene_dir.resolve()
    if a.stage=="full":
        a.observation_budget=40960000; a.evaluation_interval=8192000
        a.seeds=[0,1,2,3,4];a.repeats=3
    if not a.output_dir.exists(): run(a)
    else: p.error("output directory already exists; choose a new archive path")


if __name__=="__main__": main()
