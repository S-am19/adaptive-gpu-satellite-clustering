"""Quality trajectories and explicit censored-target summaries (no interpolation)."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def plot(folder,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder,output=Path(folder),Path(output);output.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(folder/'measurements.csv')
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[]
    for (seed,method),g in df.groupby(['seed','method']):
        good=g[g.target_satisfied]
        rows.append(dict(seed=seed,method=method,repeats=len(g),target_reached=len(good),
            median_target_complete_s=float(good.time_to_target_s.median()) if len(good)==len(g) else None,
            median_success_only_s=float(good.time_to_target_s.median()) if len(good) else None,
            median_wall_s=float(g.wall_s.median()),median_final_sse=float(g.final_exact_sse.median()),
            median_quality_overhead_s=float(g.quality_evaluation_s.median())))
    pd.DataFrame(rows).to_csv(output/'target_summary.csv',index=False)
    raw=[json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
    for seed in sorted(df.seed.unique()):
        fig,ax=plt.subplots()
        for r in raw:
            if r['seed']!=seed or r['phase']!='measured' or r['repeat']!=0:continue
            h=r['metadata']['quality_history']
            ax.plot([p['elapsed_s'] for p in h],[p['full_data_sse'] for p in h],marker='.',label=r['method'])
        target=next(r['target_sse'] for r in manifest['references'] if r['seed']==seed)
        ax.axhline(target,color='black',linestyle='--',label='Frozen target')
        ax.set(xlabel='Elapsed instrumented fit seconds',ylabel='Exact full-data SSE',title=f'Seed {seed}: measured repeat 0; no interpolation')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/f'quality_seed{seed}.png');plt.close(fig)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();plot(a.study,a.output_dir)
