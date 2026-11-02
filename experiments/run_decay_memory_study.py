#!/usr/bin/env python3
"""Development then validation study of fixed/adaptive decayed CUDA Mini-Batch.

Outputs are written incrementally to an isolated dated archive. Evaluation
targets are loaded only by this runner and are never passed to fit/controller
functions. Requires a real CUDA device; simulator and CPU fallback are rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pcap.adaptive_v2_controller import AdaptiveV2Policy
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.kmeans_cuda import fit_cuda
from pcap.minibatch_decayed_cuda import (
    fit_adaptive_decayed_minibatch_cuda,
    fit_decayed_minibatch_cuda,
)
from experiments.decay_memory_study import (
    decay_factor,
    expected_retention,
    within_batch_oldest_to_newest_weight_ratio,
)


ARCHIVE = Path(os.environ.get("PCAP_DECAY_ARCHIVE", ROOT / "outputs/20261027-decay-memory-study")).resolve()
N = 262144
D = 4
K = 16
BUDGET = 40_960_000
CHECKPOINTS = tuple(range(8_192_000, BUDGET + 1, 8_192_000))
FIXED_BATCHES = (8192, 16384, 32768, 65536)
METHODS = tuple(f"FIXED_DECAY_B{b}" for b in FIXED_BATCHES) + ("ADAPTIVE_V2", "G0_EXACT_CUDA")
DEV_X = ROOT / "outputs/20261024-real-maastricht-512/S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE/X.npy"
DEV_META = DEV_X.with_name("metadata.json")
VAL_X = ROOT / "outputs/20261025-independent-holdouts/validation512/X.npy"
VAL_META = VAL_X.with_name("metadata.json")
VAL_REFERENCE = ROOT / "outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/full_disjoint_probe/reference_runs.csv"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    existing = path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if existing:
        with path.open(newline="") as f:
            fields = next(csv.reader(f))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def _hardware():
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") not in ("0", ""):
        raise RuntimeError("CUDA simulator is enabled; refusing non-real-GPU execution")
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Real CUDA is unavailable; no CPU/simulator fallback is allowed")
    device = cuda.get_current_device()
    query = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,pstate,temperature.gpu,utilization.gpu,clocks.gr,clocks.mem", "--format=csv,noheader"]
    result = subprocess.run(query, check=True, capture_output=True, text=True)
    return {
        "nvidia_smi": result.stdout.strip(),
        "cuda_device": str(device.name),
        "compute_capability": list(device.compute_capability),
        "cuda_simulator": bool(cuda.config.ENABLE_CUDASIM),
    }


def _git_info():
    def run(args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return {"commit": run(["rev-parse", "HEAD"]), "status": run(["status", "--short", "--branch"])}


def _packages():
    from importlib.metadata import PackageNotFoundError, version
    names = ("numpy", "numba", "numba-cuda", "llvmlite", "cuda-bindings", "cuda-core", "pytest", "rasterio")
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _write_status(stage: Path, status: dict):
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(status)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _json(manifest_path, manifest)


def _write_all(stage: Path, tables: dict[str, list[dict]]):
    for name, rows in tables.items():
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            with (stage / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)


def _read_targets():
    with VAL_REFERENCE.open(newline="") as f:
        rows = {int(r["seed"]): r for r in csv.DictReader(f)}
    return {s: float(rows[s]["target_sse"]) for s in range(5)}, rows


def _fit_one(method, X, C0, seed, budget, target_sse, reference_sse, checkpoints, warm=False):
    b = 8192 if method == "ADAPTIVE_V2" else int(method.rsplit("B", 1)[1]) if method.startswith("FIXED") else None
    if method == "G0_EXACT_CUDA":
        cfg = FitConfig(k=K, seed=seed, max_steps=500, tolerance=1e-6)
        started = perf_counter()
        result = fit_cuda(X, C0, cfg)
        external = perf_counter() - started
        history = result.history
        target_observation = next((int(r.get("points_processed", 0)) for r in history
                                   if float(r.get("sse", float("inf"))) <= target_sse), None)
        target_time = external if result.sse <= target_sse else None
        target_time_scope = "fit-completion upper bound; per-iteration wall timestamps unavailable" if target_time is not None else "not reached"
        return result, external, target_observation, target_time, target_time_scope, b

    steps = (budget + b - 1) // b if method.startswith("FIXED") else budget // 256 + 1
    cfg = FitConfig(k=K, seed=seed, max_steps=steps, tolerance=0.0,
                    batch_size=b, probe_interval=20)
    kwargs = dict(observation_budget=budget, quality_interval=0,
                  quality_observation_checkpoints=checkpoints)
    if method == "ADAPTIVE_V2":
        policy = AdaptiveV2Policy.for_run(len(X), b, minimum=256, maximum=65536)
        call = lambda: fit_adaptive_decayed_minibatch_cuda(X, C0, cfg, policy=policy, **kwargs)
    else:
        call = lambda: fit_decayed_minibatch_cuda(X, C0, cfg, **kwargs)
    started = perf_counter()
    result = call()
    external = perf_counter() - started
    qhist = result.metadata.get("quality_history", [])
    first = next((q for q in qhist if float(q["full_data_sse"]) <= target_sse), None)
    target_observation = int(first["points_processed"]) if first else None
    target_time = float(first["elapsed_s"]) if first else None
    target_time_scope = "first observed at common exact full-data checkpoint" if first else "not reached at scheduled checkpoints"
    return result, external, target_observation, target_time, target_time_scope, b


def _plot(stage: Path, checkpoints, individual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["method"] for r in individual})
    good = [r for r in individual if r["method"] != "G0_EXACT_CUDA"]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in names:
        vals = [float(r["wall_s"]) for r in individual if r["method"] == method]
        sses = [float(r["final_exact_sse"]) for r in individual if r["method"] == method]
        if vals:
            ax.scatter(np.median(vals), np.median(sses), label=method, s=55)
    ax.set(xlabel="Complete fit wall time (s)", ylabel="Final exact full-data SSE", title="Runtime–quality trade-off")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(stage / "runtime_sse.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in sorted({r["method"] for r in checkpoints}):
        rows = [r for r in checkpoints if r["method"] == method]
        by_obs = {}
        for r in rows:
            by_obs.setdefault(int(r["points_processed"]), []).append(float(r["full_data_sse"]))
        if by_obs:
            xs=sorted(by_obs); ys=[np.median(by_obs[x]) for x in xs]
            ax.plot(np.array(xs)/1e6, ys, marker="o", label=method)
    ax.set(xlabel="Processed observations (millions)", ylabel="Exact full-data SSE", title="SSE at common observation checkpoints")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(stage / "sse_observations.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in names:
        rows = [r for r in individual if r["method"] == method and r.get("batch_schedule")]
        if rows:
            schedule=json.loads(rows[0]["batch_schedule"])
            ax.step(np.cumsum([0]+[int(q["batch_size"]) for q in schedule[:-1]])/1e6,
                    [int(q["batch_size"]) for q in schedule], where="post", label=method)
    ax.set(xlabel="Processed observations (millions)", ylabel="Batch size", title="Representative batch schedules")
    ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(stage / "batch_schedules.png", dpi=180); plt.close(fig)


def _summary(rows):
    import statistics
    out=[]
    for method in sorted({r["method"] for r in rows}):
        sub=[r for r in rows if r["method"]==method]
        walls=[float(r["wall_s"]) for r in sub]
        sses=[float(r["final_exact_sse"]) for r in sub]
        obs=[int(r["observations_processed"]) for r in sub]
        out.append(dict(method=method,n=len(sub),median_wall_s=statistics.median(walls),
                        wall_iqr_s=float(np.percentile(walls,75)-np.percentile(walls,25)),
                        median_final_exact_sse=statistics.median(sses),
                        sse_iqr=float(np.percentile(sses,75)-np.percentile(sses,25)),
                        target_hits=sum(str(r["target_hit"]).lower()=="true" for r in sub),
                        target_hit_rate=sum(str(r["target_hit"]).lower()=="true" for r in sub)/len(sub),
                        median_observations=int(statistics.median(obs)),
                        median_steps=float(statistics.median(int(r["steps"]) for r in sub)),
                        termination_reasons=";".join(sorted({r["termination_reason"] for r in sub}))))
    return out


def run(stage_name: str, repeats: int | None = None):
    stage = ARCHIVE / stage_name
    if stage.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {stage}")
    stage.mkdir(parents=True)
    hardware = _hardware()
    dev = stage_name.startswith("phase1_development")
    x_path, meta_path = (DEV_X, DEV_META) if dev else (VAL_X, VAL_META)
    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Prepared scene missing: {x_path}")
    X = np.load(x_path)
    scene = json.loads(meta_path.read_text())
    if X.shape != (N, D) or X.dtype != np.float32 or not X.flags.c_contiguous or not np.isfinite(X).all():
        raise ValueError(f"Expected finite contiguous float32 [{N},{D}], got {X.shape}/{X.dtype}")
    x_hash = _sha_array(X)
    if x_hash != scene["X_sha256"]:
        raise ValueError("Prepared feature hash does not match metadata")
    seeds = (0, 1, 2) if dev else (0, 1, 2, 3, 4)
    measured_repeats = 1 if dev else (3 if repeats is None else repeats)
    budget = BUDGET
    checkpoints = CHECKPOINTS
    _json(stage / "config.json", {
        "scene": scene["scene_id"], "scene_split": scene.get("split"), "N": N, "D": D, "K": K,
        "X_sha256": x_hash, "seeds": seeds, "measured_repetitions": measured_repeats,
        "warmup_fits_per_method": 2, "observation_budget": budget,
        "quality_checkpoints": checkpoints, "fixed_batch_sizes": FIXED_BATCHES,
        "adaptive_initial_batch": 8192, "adaptive_min_batch": 256, "adaptive_max_batch": 65536,
        "decay_half_life_passes": 10,
        "decay_equation": "lambda_b=0.5**(b/(10*N)); S<-lambda_b*S+batch_sum; W<-lambda_b*W+batch_count",
        "sampling": "existing per-update uniform without-replacement NumPy RNG seeded by FitConfig.seed; repeated independent draws across updates; indices hashed per update",
        "probe": "Adaptive V2 fixed 1024-row probe from SeedSequence([seed,1]); quality checkpoints exact full-data SSE; target not passed to fit/controller",
        "fit_tolerance": 0.0, "probe_interval": 20,
        "evaluation_target": "dev pilot: 5% over its G0 exact CUDA SSE for descriptive evaluation only; validation: frozen target_sse from prior full_disjoint_probe/reference_runs.csv (exact CPU SSE * 1.05)",
        "methods": METHODS,
        "method_order": "shuffled within (seed,repeat) using an experiment-only RNG; same method order across stages generated from fixed master seed 20261027",
        "G0_time_to_target": "reported as fit-completion upper bound; exact fit history has SSE per iteration but no per-iteration wall timestamps",
    })
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"complete":False,
              "stage":"development_pilot" if dev else "validation_confirmatory",
              "git":_git_info(),"hardware_before":hardware,"environment":{"python":sys.version,
              "platform":platform.platform(),"packages":_packages()},
              "inputs":{"X_path":str(x_path.relative_to(ROOT)),"X_file_sha256":_sha_file(x_path),
                        "X_array_sha256":x_hash,"metadata_path":str(meta_path.relative_to(ROOT)),
                        "metadata_sha256":_sha_file(meta_path)},
              "source_hashes":{str(p.relative_to(ROOT)):_sha_file(p) for p in [Path(__file__),ROOT/'experiments/decay_memory_study.py',ROOT/'src/pcap/minibatch_decayed_cuda.py',ROOT/'src/pcap/adaptive_v2_controller.py',ROOT/'src/pcap/minibatch_cuda.py',ROOT/'src/pcap/kmeans_cuda.py']},
              "target_visibility":"target is evaluation-only and never passed to fitting or policy functions",
              "runtime_policy":"incremental CSV checkpointing; failures preserved; no fallback"}
    _json(stage/"manifest.json",manifest)
    targets, refs = _read_targets() if not dev else ({}, {})
    reference_sse={}
    order_rng=random.Random(20261027)
    centroids={s:initialize_centroids(X,K,seed=s) for s in seeds}
    for s,c in centroids.items():
        np.savez_compressed(stage/f"initial_centroids_seed{s}.npz",C0=c)
    rows=[]; cps=[]; events=[]; timings=[]; warmrows=[]
    if dev:
        # Two bounded fits per method compile/warm all CUDA code paths without
        # putting compilation into the measured development observations.
        for method in METHODS:
            for wi in range(2):
                seed=0; C0=centroids[seed]
                warm_budget=min(BUDGET, 2*(8192 if method in ("ADAPTIVE_V2","G0_EXACT_CUDA") else int(method.rsplit("B",1)[1])))
                target=0.0; ref=0.0
                start=perf_counter()
                if method=="G0_EXACT_CUDA":
                    result=fit_cuda(X,C0,FitConfig(k=K,seed=seed,max_steps=2,tolerance=0.0))
                else:
                    b=8192 if method=="ADAPTIVE_V2" else int(method.rsplit("B",1)[1])
                    cfg=FitConfig(k=K,seed=seed,max_steps=2,tolerance=0.0,batch_size=b,probe_interval=20)
                    kw=dict(observation_budget=warm_budget,quality_interval=0)
                    if method=="ADAPTIVE_V2":
                        pol=AdaptiveV2Policy.for_run(N,b,minimum=256,maximum=65536)
                        result=fit_adaptive_decayed_minibatch_cuda(X,C0,cfg,policy=pol,**kw)
                    else: result=fit_decayed_minibatch_cuda(X,C0,cfg,**kw)
                warmrows.append(dict(method=method,warmup=wi+1,seed=seed,wall_s=perf_counter()-start,
                                     observations_processed=result.metadata.get("points_processed",0)))
        _append_csv(stage/"warmups.csv",warmrows)
    else:
        if not (ARCHIVE/"frozen_validation_protocol.json").exists():
            raise FileNotFoundError("Validation is blocked until development pilot is analyzed and frozen_validation_protocol.json is written")
        frozen=json.loads((ARCHIVE/"frozen_validation_protocol.json").read_text())
        if frozen["validation_target_source"]!=str(VAL_REFERENCE.relative_to(ROOT)):
            raise ValueError("Frozen protocol target source differs from authorized source")
        targets = {int(k):float(v) for k,v in frozen["targets_by_seed"].items()}
        if tuple(sorted(targets)) != seeds:
            raise ValueError("Frozen validation targets must cover seeds 0-4")
        for method in METHODS:
            for wi in range(2):
                seed=0; C0=centroids[seed]; target=targets[seed]
                start=perf_counter()
                if method=="G0_EXACT_CUDA": result=fit_cuda(X,C0,FitConfig(k=K,seed=seed,max_steps=500,tolerance=1e-6))
                else:
                    b=8192 if method=="ADAPTIVE_V2" else int(method.rsplit("B",1)[1])
                    cfg=FitConfig(k=K,seed=seed,max_steps=(BUDGET//256+1 if method=="ADAPTIVE_V2" else BUDGET//b),tolerance=0.0,batch_size=b,probe_interval=20)
                    kw=dict(observation_budget=min(2*max(8192,b),BUDGET),quality_interval=0)
                    if method=="ADAPTIVE_V2":
                        pol=AdaptiveV2Policy.for_run(N,b,minimum=256,maximum=65536)
                        result=fit_adaptive_decayed_minibatch_cuda(X,C0,cfg,policy=pol,**kw)
                    else: result=fit_decayed_minibatch_cuda(X,C0,cfg,**kw)
                warmrows.append(dict(method=method,warmup=wi+1,seed=seed,wall_s=perf_counter()-start,
                                     observations_processed=result.metadata.get("points_processed",0)))
        _append_csv(stage/"warmups.csv",warmrows)
    # Establish a same-data, same-C0 exact CUDA reference before randomizing
    # measured method order. The evaluation threshold remains separate.
    reference_rows=[]
    for seed in seeds:
        c0=centroids[seed]
        ref_start=perf_counter()
        ref_result=fit_cuda(X,c0,FitConfig(k=K,seed=seed,max_steps=500,tolerance=1e-6))
        ref_wall=perf_counter()-ref_start
        reference_sse[seed]=float(ref_result.sse)
        reference_rows.append({"seed":seed,"C0_sha256":_sha_array(c0),"exact_cuda_sse":float(ref_result.sse),
                               "exact_cuda_wall_s":ref_wall,"exact_cuda_steps":ref_result.metadata.get("steps"),
                               "termination_reason":ref_result.metadata.get("termination_reason"),
                               "frozen_target_sse":None if dev else targets[seed],
                               "target_rule":"development exact CUDA * 1.05 (pilot only)" if dev else "pre-existing frozen exact CPU reference * 1.05"})
        _append_csv(stage/"reference_runs.csv",[reference_rows[-1]])
    _write_status(stage,{"reference_sse_by_seed":reference_sse,"completed_reference_seeds":len(reference_rows)})
    if not dev:
        for s in seeds:
            if refs[s]["C0_sha256"]!=_sha_array(centroids[s]): raise ValueError(f"C0 hash mismatch for frozen seed {s}")
    run_idx=0
    for seed in seeds:
        for rep in range(measured_repeats):
            method_order=list(METHODS); order_rng.shuffle(method_order)
            for method in method_order:
                run_idx+=1; C0=centroids[seed]
                if method=="G0_EXACT_CUDA":
                    cfg=FitConfig(k=K,seed=seed,max_steps=500,tolerance=1e-6)
                    target=0.0 if dev else targets[seed]
                    kwargs={}
                else:
                    b=8192 if method=="ADAPTIVE_V2" else int(method.rsplit("B",1)[1])
                    cfg=FitConfig(k=K,seed=seed,max_steps=(BUDGET//256+1 if method=="ADAPTIVE_V2" else BUDGET//b),tolerance=0.0,batch_size=b,probe_interval=20)
                    target=0.0 if dev else targets[seed]
                    kwargs={"observation_budget":BUDGET,"quality_interval":0,"quality_observation_checkpoints":CHECKPOINTS}
                    if method=="ADAPTIVE_V2": kwargs["policy"]=AdaptiveV2Policy.for_run(N,b,minimum=256,maximum=65536)
                started=perf_counter()
                try:
                    if method=="G0_EXACT_CUDA": result=fit_cuda(X,C0,cfg)
                    elif method=="ADAPTIVE_V2": result=fit_adaptive_decayed_minibatch_cuda(X,C0,cfg,**kwargs)
                    else: result=fit_decayed_minibatch_cuda(X,C0,cfg,**kwargs)
                    external=perf_counter()-started
                    if not np.isfinite(result.sse) or not np.isfinite(result.centroids).all(): raise FloatingPointError("nonfinite output")
                    ref=reference_sse.get(seed)
                    target=1.05*ref if dev and ref is not None else target
                    sse=float(result.sse); hit=bool(sse<=target) if target else None
                    steps=int(result.metadata.get("steps",len(result.history)))
                    if method=="G0_EXACT_CUDA":
                        # Exact Lloyd has no Mini-Batch points_processed field.
                        # Each update assigns all N points; the fit also makes
                        # one initial and one final assignment pass.
                        obs=steps*N
                        assignment_points=(steps+1)*N
                    else:
                        obs=int(result.metadata.get("points_processed",0))
                        assignment_points=result.metadata.get("assignment_points_processed")
                    schedule=[{"step":int(q.get("step",i+1)),"batch_size":int(q.get("batch_size",q.get("active_batch_size",0))),
                               "next_batch_size":q.get("next_batch_size"),"points_processed":int(q.get("points_processed",0)),
                               "sample_indices_sha256":q.get("sample_indices_sha256"),
                               "centroid_displacement":q.get("centroid_displacement"),
                               "probe_sse":q.get("probe_sse"),"probe_label_change_fraction":q.get("probe_label_change_fraction"),
                               "controller_decision":q.get("controller_decision")} for i,q in enumerate(result.history)]
                    row=dict(run_id=run_idx,method=method,seed=seed,repeat=rep+1,device=result.metadata.get("execution_device",result.metadata.get("algorithm")),
                             N=N,D=D,K=K,X_sha256=x_hash,C0_sha256=_sha_array(C0),target_sse=target,
                             target_evaluation_only=True,target_hit=hit,final_exact_sse=sse,
                             relative_gap_to_exact_reference=(sse/ref-1 if ref else None),
                             wall_s=external,observations_processed=obs,steps=steps,
                             assignment_points_processed=assignment_points,
                             termination_reason=result.metadata.get("termination_reason"),
                             convergence_satisfied=result.metadata.get("convergence_satisfied"),
                             first_target_crossing_observations=(
                                 next((int(r.get("points_processed",0)) for r in result.history
                                       if float(r.get("sse",float("inf")))<=target),None)
                                 if method=="G0_EXACT_CUDA" and target else None),
                             first_target_crossing_time_s=None,
                             target_time_scope="G0 fit-completion upper bound" if method=="G0_EXACT_CUDA" else "common exact full-data SSE checkpoint",
                             time_to_target_s=(external if method=="G0_EXACT_CUDA" and hit else None),
                             batch_transitions=(sum(1 for q in schedule if q.get("next_batch_size") is not None and
                                                    q.get("batch_size")!=q.get("next_batch_size")) if method=="ADAPTIVE_V2" else 0),
                             batch_schedule=json.dumps(schedule,separators=(",",":")),
                             effective_cluster_counts=json.dumps(result.metadata.get("effective_cluster_counts",result.metadata.get("cluster_observation_counts",[])),separators=(",",":")),
                             decay_half_life_observations=10*N if method!="G0_EXACT_CUDA" else None,
                             theoretical_old_history_retention=expected_retention(obs,N) if method!="G0_EXACT_CUDA" else None,
                             timing_json=json.dumps(result.timings,sort_keys=True),metadata_json=json.dumps(result.metadata,sort_keys=True,default=str))
                    # Time-to-target for Mini-Batch is read from actual checkpoint
                    # timestamps; G0 has no per-iteration wall timestamps.
                    if method!="G0_EXACT_CUDA" and target:
                        q=next((q for q in result.metadata.get("quality_history",[]) if float(q.get("full_data_sse",float("inf")))<=target),None)
                        if q:
                            row["first_target_crossing_observations"]=int(q["points_processed"])
                            row["first_target_crossing_time_s"]=float(q["elapsed_s"])
                    rows.append(row)
                    _append_csv(stage/"individual_runs.csv",[row])
                    if method!="G0_EXACT_CUDA":
                        for q in result.metadata.get("quality_history",[]):
                            cp={"method":method,"seed":seed,"repeat":rep+1,"step":q.get("step"),"elapsed_s":q.get("elapsed_s"),
                                "full_data_sse":q.get("full_data_sse"),"points_processed":q.get("points_processed"),
                                "target_sse_evaluation_only":target,"target_hit_at_checkpoint":float(q.get("full_data_sse",float("inf")))<=target if target else None,
                                "observation_budgets_reached":json.dumps(q.get("observation_budgets_reached",[]))}
                            cps.append(cp); _append_csv(stage/"checkpoints.csv",[cp])
                    for q in result.history:
                        ev=dict(method=method,seed=seed,repeat=rep+1,step=q.get("step"),batch_size=q.get("batch_size"),
                                next_batch_size=q.get("next_batch_size"),points_processed=q.get("points_processed"),
                                sample_indices_sha256=q.get("sample_indices_sha256"),centroid_displacement=q.get("centroid_displacement"),
                                forgetting_factor=q.get("forgetting_factor"),probe_sse=q.get("probe_sse"),
                                normalized_checkpoint_displacement=q.get("normalized_checkpoint_displacement"),
                                probe_label_change_fraction=q.get("probe_label_change_fraction"),
                                controller_decision=json.dumps(q.get("controller_decision"),sort_keys=True) if q.get("controller_decision") is not None else None,
                                adaptation_frozen=q.get("adaptation_frozen"),convergence_satisfied=q.get("convergence_satisfied"))
                        events.append(ev); _append_csv(stage/"controller_events.csv",[ev])
                    timing={"method":method,"seed":seed,"repeat":rep+1,"external_wall_s":external}
                    timing.update({f"internal_{k}":v for k,v in result.timings.items()})
                    timings.append(timing); _append_csv(stage/"timing_breakdown.csv",[timing])
                    np.savez_compressed(stage/f"centroids_{method}_seed{seed}_rep{rep+1}.npz",centroids=result.centroids)
                except Exception as exc:
                    failure={"run_id":run_idx,"method":method,"seed":seed,"repeat":rep+1,"error":repr(exc),"traceback":traceback.format_exc(),"utc":datetime.now(timezone.utc).isoformat()}
                    with (stage/"failures.jsonl").open("a") as f: f.write(json.dumps(failure)+"\n")
                    _write_status(stage,{"complete":False,"failure":failure,"completed_runs":len(rows)})
                    raise
                _write_status(stage,{"complete":False,"completed_runs":len(rows),"last_run_id":run_idx})
    summaries=_summary(rows)
    _write_all(stage,{"method_summary.csv":summaries})
    _plot(stage,cps,rows)
    _write_status(stage,{"complete":True,"completed_runs":len(rows),"failed_runs":0,
                         "reference_exact_sse_by_seed":reference_sse,
                         "gpu_conditions_after":_hardware()["nvidia_smi"]})
    checksum=stage/"checksums.sha256"
    entries=[]
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path!=checksum:
            entries.append(f"{_sha_file(path)}  {path.relative_to(stage)}")
    checksum.write_text("\n".join(entries)+"\n")
    return stage


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("phase1_development","phase1_development_retry1","phase2_validation"),required=True)
    ap.add_argument("--repeats",type=int,default=None)
    ap.add_argument("--archive-root",type=Path,default=None,
                    help="new output root; defaults to PCAP_DECAY_ARCHIVE or the dated study path")
    args=ap.parse_args()
    global ARCHIVE
    if args.archive_root is not None: ARCHIVE=args.archive_root.resolve()
    path=run(args.stage,args.repeats)
    print(path)


if __name__=="__main__": main()
