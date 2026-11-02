"""Per-seed external fit medians/IQR and matched exact CPU/CUDA speedups."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(csv_path, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(csv_path)
    identity=[c for c in ('dataset_id','case_id','session_id') if c in df]
    keys=identity+['method','implementation','n','d','k','seed']
    rows=[]
    for key,group in df.groupby(keys):
        t=group.wall_time_s.to_numpy()
        row=dict(zip(keys,key))
        row.update(repeats=len(t),median_fit_s=float(np.median(t)),
                   q25_fit_s=float(np.percentile(t,25)),q75_fit_s=float(np.percentile(t,75)),
                   iqr_fit_s=float(np.percentile(t,75)-np.percentile(t,25)),
                   min_fit_s=float(t.min()),max_fit_s=float(t.max()),
                   median_sse=float(group.final_exact_sse.median()),
                   steps_min=int(group.steps.min()),steps_max=int(group.steps.max()),
                   termination_reasons=';'.join(sorted(group.termination_reason.unique())),
                   converged_repeats=int(group.convergence_satisfied.sum()),
                   median_observations=float(group.points_processed.median()))
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(output_dir/'median_summary.csv',index=False)
    pairs=[]
    for key,group in summary.groupby(identity+['n','d','k','seed']):
        exact=group.set_index('method')
        if 'C0' in exact.index and 'G0' in exact.index:
            cpu,gpu=exact.loc['C0'],exact.loc['G0']
            valid=(cpu.steps_min==cpu.steps_max==gpu.steps_min==gpu.steps_max and
                   cpu.termination_reasons==gpu.termination_reasons and
                   np.isclose(cpu.median_sse,gpu.median_sse,rtol=1e-5,atol=1e-8))
            pairs.append(dict(zip(identity+['n','d','k','seed'],key),matched_work=valid,
                              speedup=float(cpu.median_fit_s/gpu.median_fit_s) if valid else np.nan))
    pd.DataFrame(pairs).to_csv(output_dir/'exact_speedups.csv',index=False)
    for key,group in summary.groupby(identity+['d','k','seed']):
        fig,ax=plt.subplots()
        for (method,implementation), sub in group.groupby(['method','implementation']):
            sub=sub.sort_values('n')
            ax.errorbar(sub.n,sub.median_fit_s,
                yerr=np.array([sub.median_fit_s-sub.q25_fit_s,sub.q75_fit_s-sub.median_fit_s]),
                marker='o',label=f'{method}: {implementation}')
        ax.set(xlabel='N (points)',ylabel='External complete-fit seconds',yscale='log',xscale='log',
               title=f'D={key[-3]}, K={key[-2]}, seed={key[-1]}')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output_dir/f'scaling_{"_".join(str(v) for v in key)}.png');plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();summarize(args.csv,args.output_dir)
