#!/usr/bin/env python3
"""Trace-only matched cumulative-vs-decayed diagnostic on 31UFU validation."""

import csv
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from pcap.budgeted_minibatch import fit_budgeted_minibatch
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.minibatch_decayed_cuda import fit_decayed_minibatch_cuda

OUT=ROOT/"outputs/20261027-decay-memory-study/phase0_convergence_diagnostic_retry1"
XPATH=ROOT/"outputs/20261025-independent-holdouts/validation512/X.npy"
METAPATH=XPATH.with_name("metadata.json")
DECAY_RUNS=ROOT/"outputs/20261027-decay-memory-study/phase2_validation/individual_runs.csv"
N,D,K,BUDGET=262144,4,16,40_960_000
CHECKPOINTS=tuple(range(8_192_000,BUDGET+1,8_192_000))

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,default=OUT)
    args=ap.parse_args()
    OUT=args.output_dir.resolve()
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():raise RuntimeError("real CUDA required")
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total,memory.free,pstate,temperature.gpu,utilization.gpu,clocks.gr,clocks.mem","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.strip()
    X=np.load(XPATH); meta=json.loads(METAPATH.read_text())
    if X.shape!=(N,D) or X.dtype!=np.float32 or arrsha(X)!=meta["X_sha256"]:raise ValueError("prepared validation X mismatch")
    (OUT/"config.json").write_text(json.dumps({"scene":meta["scene_id"],"N":N,"D":D,"K":K,"seeds":[0,1,2],"batch_size":8192,"observation_budget":BUDGET,"checkpoints":CHECKPOINTS,"probe_policy":"overlapping fixed 1024-row probe; identical population/sampler as decayed baseline","diagnostic_trace":True,"timing_use":"diagnostic only; host copies added for counts/centroid checkpoints; do not compare runtime with uninstrumented runs","decay_policy":"existing blockwise observation-scaled half-life 10N"},indent=2)+"\n")
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"complete":False,"device":gpu,"cuda_simulator":False,"python":sys.version,"platform":platform.platform(),"git_head":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"inputs":{"X":str(XPATH.relative_to(ROOT)),"X_file_sha256":sha(XPATH),"X_array_sha256":arrsha(X),"metadata_sha256":sha(METAPATH)},"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"src/pcap/budgeted_minibatch.py",ROOT/"src/pcap/minibatch_decayed_cuda.py"]},"runs_completed":0}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    c0s={s:initialize_centroids(X,K,seed=s) for s in range(3)}
    for s,c in c0s.items():np.savez_compressed(OUT/f"C0_seed{s}.npz",C0=c)
    csv.field_size_limit(64*1024*1024)
    refrows=list(csv.DictReader(DECAY_RUNS.open()))
    decay_hashes={}
    event_rows=list(csv.DictReader((DECAY_RUNS.parent/"controller_events.csv").open()))
    for s in range(3):
        decay_hashes[s]=[r["sample_indices_sha256"] for r in event_rows if r.get("method")=="FIXED_DECAY_B8192" and int(r["seed"])==s and int(r["repeat"])==1 and r.get("sample_indices_sha256")]
    # Two small warm-ups per path; these data are excluded from reports.
    warm=[]
    for method in ("cumulative","decayed"):
        for w in range(2):
            cfg=FitConfig(k=K,seed=0,max_steps=2,tolerance=0,batch_size=8192,probe_interval=20)
            t=perf_counter()
            if method=="cumulative":
                r,_,_=fit_budgeted_minibatch(X,c0s[0],cfg,device="cuda",observation_budget=16384,evaluation_interval=8192,disjoint_probe=False,diagnostic_trace=True)
            else:r=fit_decayed_minibatch_cuda(X,c0s[0],cfg,observation_budget=16384,quality_interval=0)
            warm.append({"method":method,"warmup":w+1,"wall_s":perf_counter()-t})
    with (OUT/"warmups.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["method","warmup","wall_s"]);wr.writeheader();wr.writerows(warm)
    runs=[]; checkpoints=[]; updates=[]; paired=[]
    for seed in range(3):
        c0=c0s[seed]
        cfg=FitConfig(k=K,seed=seed,max_steps=BUDGET//8192,tolerance=0,batch_size=8192,probe_interval=20)
        started=perf_counter()
        cum,cps,_=fit_budgeted_minibatch(X,c0,cfg,device="cuda",observation_budget=BUDGET,evaluation_interval=8_192_000,disjoint_probe=False,diagnostic_trace=True)
        cum_wall=perf_counter()-started
        started=perf_counter()
        dec=fit_decayed_minibatch_cuda(X,c0,cfg,quality_interval=0,observation_budget=BUDGET,quality_observation_checkpoints=CHECKPOINTS)
        dec_wall=perf_counter()-started
        cum_hash=[r.get("sample_indices_sha256") for r in cum.history]
        dec_hash=[r.get("sample_indices_sha256") for r in dec.history]
        matched=cum_hash==dec_hash==decay_hashes[seed]
        if not matched:raise RuntimeError(f"Sampling streams did not match for seed {seed}")
        for method,result,wall in (("CUMULATIVE",cum,cum_wall),("DECAYED_FIXED_8192",dec,dec_wall)):
            runs.append({"method":method,"seed":seed,"N":N,"D":D,"K":K,"X_sha256":arrsha(X),"C0_sha256":arrsha(c0),"wall_s_diagnostic_only":wall,"final_exact_sse":result.sse,"observations_processed":result.metadata["points_processed"],"steps":result.metadata["steps"],"termination_reason":result.metadata["termination_reason"],"convergence_satisfied":result.metadata["convergence_satisfied"],"sampling_stream_matches":matched,"cluster_counts_or_effective_counts":json.dumps(result.metadata.get("cluster_observation_counts",result.metadata.get("effective_cluster_counts",[]))),"timings":json.dumps(result.timings,sort_keys=True)})
            for h in result.history:
                updates.append({"method":method,"seed":seed,"step":h.get("step"),"batch_size":h.get("actual_batch_size",h.get("batch_size")),"points_processed":h.get("points_processed"),"sample_indices_sha256":h.get("sample_indices_sha256"),"centroid_displacement":h.get("centroid_displacement"),"probe_sse":h.get("probe_sse"),"probe_label_change_fraction":h.get("probe_label_change_fraction")})
        for q in cps:
            checkpoints.append({"method":"CUMULATIVE","seed":seed,**q})
        for q in dec.metadata["quality_history"]:
            if q.get("points_processed") in CHECKPOINTS:
                checkpoints.append({"method":"DECAYED_FIXED_8192","seed":seed,**q})
        for i,(a,b) in enumerate(zip(cps,[q for q in dec.metadata["quality_history"] if q.get("points_processed") in CHECKPOINTS])):
            paired.append({"seed":seed,"points_processed":a["points_processed"],"cumulative_sse":a["full_exact_sse"],"decayed_sse":b["full_data_sse"],"decayed_minus_cumulative_sse":b["full_data_sse"]-a["full_exact_sse"],"cumulative_checkpoint_centroid_movement":a.get("checkpoint_centroid_movement"),"cumulative_cluster_update_counts":json.dumps(a.get("cluster_update_counts",[])),"decayed_effective_cluster_counts":json.dumps(dec.metadata.get("effective_cluster_counts",[])) if i==len(cps)-1 else None})
    def write(name,rows):
        if not rows:return
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write("individual_runs.csv",runs);write("checkpoints.csv",checkpoints);write("update_trace.csv",updates);write("paired_checkpoint_comparison.csv",paired)
    manifest.update({"complete":True,"runs_completed":len(runs),"sampling_streams_identical_to_decayed":True,"trace_note":"cumulative centroid/count snapshots add device-to-host diagnostic overhead; runtime is not a benchmark comparison","gpu_conditions_after":subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total,memory.free,pstate,temperature.gpu,utilization.gpu,clocks.gr,clocks.mem","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.strip()})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    checks=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name!="checksums.sha256":checks.append(f"{sha(p)}  {p.relative_to(OUT)}")
    (OUT/"checksums.sha256").write_text("\n".join(checks)+"\n")
    print(OUT)

if __name__=="__main__":main()
