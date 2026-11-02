"""Bounded projection-ANN vs exhaustive CUDA assignment, including index maintenance."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pcap.assignments_ann_cuda import ProjectionCUDAIndex
from pcap.assignments_exact import assign_exact
from pcap.minibatch_cuda import _assign_indexed_kernel,_blocks,_THREADS
from numba import cuda
from run_suite import provenance,snapshot_sources


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--n',type=int,default=8192)
    p.add_argument('--dimensions',type=int,nargs='+',default=[4,14])
    p.add_argument('--clusters',type=int,nargs='+',default=[4,8,16,32,64,128,256])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--budget-seconds',type=float,default=120)
    args=p.parse_args(argv)
    if min([args.n,args.repeats,args.warmups]+args.dimensions+args.clusters)<1 or args.seed<0 or args.budget_seconds<=0:p.error('Invalid settings')
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():raise RuntimeError('Real CUDA required')
    args.output_dir.mkdir(parents=True,exist_ok=False);start=perf_counter();rows=[]
    manifest=dict(provenance(),config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},complete=False,failures=[],
        scope='resident points and output buffers common outside timers; complete_s independently includes centroid/index upload, query and result download; query_s overlaps complete_s',
        index='one seeded sorted random projection, CPU construction and CUDA queries; exact_small_k is explicit')
    snapshot_sources(args.output_dir/"source",manifest["source_hashes"])
    try:
        with (args.output_dir/'raw.csv').open('x',newline='') as raw:
            writer=None
            for d in args.dimensions:
                for k in args.clusters:
                    rng=np.random.default_rng(np.random.SeedSequence([args.seed,d,k]))
                    X=rng.normal(size=(args.n,d)).astype(np.float32);C=rng.normal(size=(k,d)).astype(np.float32)
                    exact=assign_exact(X,C)
                    dx=cuda.to_device(X);indices=cuda.to_device(np.arange(len(X),dtype=np.int32))
                    labels=cuda.device_array(len(X),np.int32);distances=cuda.device_array(len(X),np.float64)
                    index=ProjectionCUDAIndex(seed=args.seed)
                    methods=[('exact',k),('projection',1),('projection',4),('projection',16)]
                    for phase,count in [('warmup',args.warmups),('measured',args.repeats)]:
                        for repeat in range(count):
                            order=methods[repeat%4:]+methods[:repeat%4]
                            for method,effort in order:
                                if perf_counter()-start>args.budget_seconds:raise TimeoutError('ANN study budget reached between queries')
                                cuda.synchronize();t=perf_counter()
                                if method=='exact':
                                    dc=cuda.to_device(C);cuda.synchronize();mode='exact';budget=k
                                else:
                                    index.prepare(C,effort);budget=index.effective_effort(effort);mode=index.search_mode(effort)
                                build_s=perf_counter()-t
                                q=perf_counter()
                                if method=='exact':
                                    _assign_indexed_kernel[_blocks(len(X)),_THREADS](dx,indices,dc,labels,distances);cuda.synchronize()
                                else:index.assign_device(dx,indices,labels,distances,version=index.version,effort=effort)
                                query_s=perf_counter()-q
                                host_labels=labels.copy_to_host();host_dist=distances.copy_to_host();cuda.synchronize()
                                complete_s=perf_counter()-t
                                independent=np.sum((X.astype(float)-C[host_labels].astype(float))**2,axis=1)
                                np.testing.assert_allclose(host_dist,independent,rtol=1e-12,atol=1e-12)
                                excess=np.maximum(host_dist-exact.squared_distances,0)
                                row=dict(n=len(X),d=d,k=k,seed=args.seed,method=method,effort=effort,effective_candidates=budget,search_mode=mode,
                                    phase=phase,repeat=repeat,build_upload_s=build_s,query_s=query_s,complete_s=complete_s,
                                    centroid_distance_evaluations=len(X)*budget,strict_recall=float(np.mean(host_labels==exact.labels)),
                                    distance_agreement=float(np.mean(np.isclose(host_dist,exact.squared_distances,rtol=1e-10,atol=1e-12))),
                                    mean_excess_squared_distance=float(excess.mean()),max_excess_squared_distance=float(excess.max()),
                                    objective_gap=float((host_dist.sum()-exact.squared_distances.sum())/max(exact.squared_distances.sum(),1e-30)))
                                if writer is None:writer=csv.DictWriter(raw,fieldnames=list(row));writer.writeheader()
                                writer.writerow(row);raw.flush();rows.append(row)
        manifest['complete']=True
    except Exception as exc:
        manifest['failures'].append(f'{type(exc).__name__}: {exc}');raise
    finally:
        manifest['raw_rows']=len(rows);(args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (args.output_dir/'checksums.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in sorted(args.output_dir.iterdir()) if f.is_file()))
    print('ANN artifacts:',args.output_dir,'rows:',len(rows))
    return args.output_dir


if __name__=='__main__':main()
