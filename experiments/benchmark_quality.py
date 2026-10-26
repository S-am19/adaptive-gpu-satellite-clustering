"""Persistent, instrumented time-to-common-quality study; explicit CPU/GPU methods.

Reference SSE is from a converged exact CPU fit (outside measured fits). Target
is reference*(1+gap), frozen before measurements. All methods pay for the same
initial/periodic/terminal full-data quality evaluations, inside fit wall time.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from pcap.contracts import FitConfig,validate_features
from pcap.initialization import initialize_centroids
from pcap.kmeans_cpu import fit_cpu
from pcap.controlled_fit import fit_controlled
from pcap.adaptive_controller import BatchPolicy
from pcap.adaptive_ann_controller import ANNEffortPolicy
from run_suite import provenance,array_hash,validate_result,snapshot_sources

MODES={'C0':('cpu','full',False),'C1':('cpu','fixed',False),
       'G0':('cuda','full',False),'G1':('cuda','fixed',False),
       'G3CUDA':('cuda','batch_only',False),'G2CUDA':('cuda','fixed',True),
       'G4CUDA':('cuda','ann_only',True),'G5CUDA':('cuda','combined',True),
       'G2':('cpu','fixed',True),'G3':('cpu','batch_only',False),
       'G4':('cpu','ann_only',True),'G5':('cpu','combined',True)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=MODES,default=['C0','G0','G1','G2CUDA','G3CUDA','G4CUDA','G5CUDA'])
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--reference-steps',type=int,default=1000)
    p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--ann-effort',type=int,default=None)
    p.add_argument('--policy-json',type=Path,help='Frozen batch/ann_cpu/ann_cuda policy settings')
    p.add_argument('--quality-gap',type=float,default=.05)
    p.add_argument('--evaluation-interval',type=int,default=10)
    p.add_argument('--probe-interval',type=int,default=10)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--budget-seconds',type=float,default=120)
    args=p.parse_args(argv)
    if (min(args.k,args.max_steps,args.reference_steps,args.batch_size,args.evaluation_interval,args.probe_interval,args.warmups,args.repeats)<1
            or min(args.seeds)<0 or not np.isfinite(args.quality_gap) or args.quality_gap<0
            or not np.isfinite(args.budget_seconds) or args.budget_seconds<=0):
        p.error('Invalid study settings')
    if args.policy_json and args.ann_effort is not None:
        p.error('--ann-effort cannot be combined with --policy-json; set initial_effort in the policy file')
    policy_settings=json.loads(args.policy_json.read_text()) if args.policy_json else None
    if policy_settings is not None and (not isinstance(policy_settings,dict)
            or set(policy_settings)-{'batch','ann_cpu','ann_cuda'}):
        p.error('Unknown policy configuration sections')
    X=np.load(args.scene_dir/'X.npy',allow_pickle=False);validate_features('X',X)
    scene=json.loads((args.scene_dir/'metadata.json').read_text());x_hash=array_hash(X)
    if x_hash!=scene.get('X_sha256'):raise ValueError('Prepared scene hash mismatch')
    if args.k>len(X):p.error('K exceeds N')
    parsed_policies={}
    if policy_settings:
        for section,options in policy_settings.items():
            if not isinstance(options,dict):p.error('Policy sections must be objects')
            if section=='batch':
                parsed_policies[section]=BatchPolicy.for_run(len(X),min(args.batch_size,len(X)),**options)
            else:
                parsed_policies[section]=ANNEffortPolicy(**options)
        json.dumps(policy_settings,allow_nan=False)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=perf_counter();records=[];references=[]
    manifest=dict(provenance(),config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  policy_settings=policy_settings,
                  scene=scene,X_sha256=x_hash,complete=False,failures=[],references=references,
                  timing_scope='instrumented persistent fits; all full-data monitoring inside fit timer; reference/warmups excluded',
                  target_policy='first measured initial/checkpoint/terminal objective <= converged reference SSE*(1+gap); no interpolation')
    snapshot_sources(args.output_dir/"source",manifest["source_hashes"])
    def budget():
        if perf_counter()-start>args.budget_seconds:raise TimeoutError('Quality study budget exhausted between fits')
    try:
        with (args.output_dir/'raw.jsonl').open('x') as raw:
            for seed in args.seeds:
                budget();C0=initialize_centroids(X,args.k,seed=seed);c_hash=array_hash(C0)
                ref=fit_cpu(X,C0,FitConfig(args.k,seed,args.reference_steps,args.tolerance))
                if ref.metadata['termination_reason']!='tolerance':
                    raise RuntimeError('Reference did not converge; do not manufacture a target from capped SSE')
                target=ref.sse*(1+args.quality_gap)
                references.append(dict(seed=seed,C0_sha256=c_hash,exact_sse=ref.sse,steps=ref.metadata['steps'],target_sse=target))
                for phase,repeats in [('warmup',args.warmups),('measured',args.repeats)]:
                    for repeat in range(repeats):
                        offset=(seed+repeat)%len(args.methods);order=args.methods[offset:]+args.methods[:offset]
                        for method in order:
                            budget();device,mode,ann=MODES[method]
                            effort_policy=parsed_policies.get('ann_cuda' if device=='cuda' else 'ann_cpu') if ann else None
                            initial_effort=(effort_policy.initial_effort if effort_policy is not None else args.ann_effort)
                            if initial_effort is None and ann:
                                initial_effort=min(args.k,16) if device=='cuda' else 64
                            cfg=FitConfig(args.k,seed,args.max_steps,args.tolerance,
                                None if mode=='full' else min(args.batch_size,len(X)),args.probe_interval,initial_effort if ann else None)
                            batch_policy=parsed_policies.get('batch') if mode in ('batch_only','combined') else None
                            ann_policy=effort_policy if mode in ('ann_only','combined') else None
                            if device=='cuda':
                                from numba import cuda
                                cuda.synchronize()
                            t=perf_counter()
                            result=fit_controlled(X,C0,cfg,device=device,mode=mode,use_ann=ann,
                                batch_policy=batch_policy,ann_policy=ann_policy,
                                evaluation_interval=args.evaluation_interval,target_sse=target)
                            if device=='cuda':cuda.synchronize()
                            elapsed=perf_counter()-t
                            validate_result(X,result)
                            if array_hash(X)!=x_hash or array_hash(C0)!=c_hash:raise ValueError('Fit mutated common inputs')
                            record=dict(method=method,execution_device=device,phase=phase,repeat=repeat,seed=seed,
                                X_sha256=x_hash,C0_sha256=c_hash,wall_s=elapsed,target_sse=target,
                                metadata=result.metadata,timings=result.timings,history=result.history)
                            raw.write(json.dumps(record,allow_nan=False)+'\n');raw.flush()
                            if phase=='measured':
                                records.append(dict(method=method,execution_device=device,seed=seed,repeat=repeat,
                                    wall_s=elapsed,time_to_target_s=elapsed if result.metadata['target_satisfied'] else None,first_observed_target_s=result.metadata['first_observed_target_s'],
                                    target_satisfied=result.metadata['target_satisfied'],target_sse=target,
                                    final_exact_sse=result.sse,steps=result.metadata['steps'],
                                    termination_reason=result.metadata['termination_reason'],
                                    points_processed=result.metadata['points_processed'],quality_evaluation_s=result.timings['quality_evaluation_s']))
        manifest['complete']=True
    except Exception as exc:
        manifest['failures'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['completed_measured_fits']=len(records)
        (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
        with (args.output_dir/'measurements.csv').open('x',newline='') as f:
            if records:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        (args.output_dir/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in sorted(args.output_dir.iterdir()) if f.is_file()))
    print(f'Quality artifacts: {args.output_dir}; measured fits: {len(records)}')
    return args.output_dir


if __name__=='__main__':main()
