"""Optional C++ OpenMP baseline; compilation is explicit, never automatic."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from time import perf_counter
import numpy as np
from .contracts import FitConfig,RunResult,validate_points_centroids,validate_integer

ROOT=Path(__file__).resolve().parents[2]


def fit_openmp(X,C0,config,*,threads=1,executable=None):
    start=perf_counter()
    if not isinstance(config,FitConfig):raise TypeError('config must be FitConfig')
    validate_points_centroids(X,C0);validate_integer('threads',threads)
    if threads>256:raise ValueError('threads exceeds bounded baseline limit 256')
    if config.batch_size is not None or config.ann_effort is not None:raise ValueError('OpenMP Lloyd requires batch_size=ann_effort=None')
    n,d=X.shape;k=len(C0)
    if config.k!=k or k>n or n>np.iinfo(np.int32).max:raise ValueError('Invalid K or indexed capacity')
    exe=Path(executable) if executable else ROOT/'outputs/build/pcap_openmp'
    if not exe.is_file():raise FileNotFoundError('Build the optional baseline with bash scripts/build_openmp.sh')
    build_file=Path(str(exe)+'.json')
    build=json.loads(build_file.read_text()) if build_file.exists() else None
    if build and (build['source_sha256']!=hashlib.sha256((ROOT/'native/kmeans_openmp.cpp').read_bytes()).hexdigest() or build['executable_sha256']!=hashlib.sha256(exe.read_bytes()).hexdigest()):
        raise RuntimeError('Stale OpenMP build; rebuild explicitly')
    payload=b'PCAPOMP1'+struct.pack('<QQQQdI',n,d,k,int(config.max_steps),float(config.tolerance),int(threads))+X.tobytes()+C0.tobytes()
    process_start=perf_counter()
    env=dict(os.environ,OMP_DYNAMIC='FALSE',OMP_NUM_THREADS=str(threads))
    result=subprocess.run([str(exe.resolve())],input=payload,capture_output=True,env=env,check=False)
    process_s=perf_counter()-process_start
    if result.returncode:raise RuntimeError('OpenMP fit failed: '+result.stderr.decode(errors='replace'))
    raw=result.stdout
    if raw[:8]!=b'PCAPOUT1':raise RuntimeError('Invalid OpenMP output protocol')
    steps,converged,actual,native_s,final_s=struct.unpack_from('<QIIdd',raw,8);offset=40
    expected=offset+4*k*d+4*n+16*(steps+1)
    if len(raw)!=expected:raise RuntimeError('Truncated OpenMP output')
    C=np.frombuffer(raw,np.float32,k*d,offset).reshape(k,d).copy();offset+=4*k*d
    labels=np.frombuffer(raw,np.int32,n,offset).copy();offset+=4*n
    objectives=np.frombuffer(raw,np.float64,steps+1,offset);offset+=8*(steps+1)
    movements=np.frombuffer(raw,np.float64,steps+1,offset)
    if not np.isfinite(C).all() or not np.isfinite(objectives).all():raise FloatingPointError('Nonfinite OpenMP result')
    history=[dict(step=i,sse=float(objectives[i]),centroid_displacement=float(movements[i]),points_processed=i*n) for i in range(steps+1)]
    meta=dict(algorithm='cpp_openmp_lloyd',execution_device='cpu',config=asdict(config),
        final_exact_sse=float(objectives[-1]),steps=int(steps),termination_reason='tolerance' if converged else 'max_steps',
        convergence_satisfied=bool(converged),threads_requested=int(threads),threads_actual=int(actual),
        points_processed=int(steps)*n,assignment_points_processed=(int(steps)+1)*n,
        source_sha256=hashlib.sha256((ROOT/'native/kmeans_openmp.cpp').read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        build_flags=build['flags'] if build else 'unverified custom executable',compiler=build['compiler'] if build else None,
        timing_scope='wall includes binary serialization/subprocess/IPC; native_fit_s excludes I/O/startup; never sum scopes',
        input_dtype='float32',accumulation_dtype='float64',empty_cluster_policy='retain',tie_policy='lowest_index')
    return RunResult(C,labels,history,dict(wall_s=perf_counter()-start,compute_s=native_s,native_fit_s=native_s,process_s=process_s,final_assignment_s=final_s),meta)
