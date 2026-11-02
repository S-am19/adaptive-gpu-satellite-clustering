"""Build aggregate tables and plots from the preserved 2026-10-31 study runs."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/20261031-workload-scalability-study"
COMPLETE_RUNS = [
    OUT / "pilot-seed0-corrected/manifest.json",
    OUT / "confirmatory-seeds1-2/N4194265-K8-16/manifest.json",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for item in items for key in item))
    if not keys:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)


def quantiles(values: list[float]) -> tuple[float, float, float]:
    return (float(np.median(values)), float(np.quantile(values, .25)), float(np.quantile(values, .75)))


def collect() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    run_rows, checkpoint_rows, batch_rows, controller_rows, sources = [], [], [], [], []
    for manifest_path in COMPLETE_RUNS:
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            continue
        base = manifest_path.parent
        sources.append({"run_directory": str(base.relative_to(ROOT)),
                        "manifest_sha256": sha(manifest_path),
                        "completed_fits": manifest.get("completed_measured_fits"),
                        "seeds": json.dumps(manifest["config"]["seeds"]),
                        "sizes": json.dumps(list(manifest["config"]["sizes"]))})
        for phase in ("phase_exact", "phase_cumulative", "phase_decayed"):
            for row in rows(base / phase / "individual_runs.csv"):
                row["source_run_directory"] = str(base.relative_to(ROOT))
                run_rows.append(row)
        for row in rows(base / "checkpoints.csv"):
            row["source_run_directory"] = str(base.relative_to(ROOT))
            checkpoint_rows.append(row)
        # The runner wrote a duplicate trajectory trace per fit. Collapse exact
        # duplicate rows in this derived view; original raw CSVs remain intact.
        for row in rows(base / "batch_trajectories.csv"):
            row["source_run_directory"] = str(base.relative_to(ROOT))
            batch_rows.append(row)
        for row in rows(base / "controller_events.csv"):
            row["source_run_directory"] = str(base.relative_to(ROOT))
            controller_rows.append(row)
    if not run_rows:
        raise RuntimeError("No complete workload-study run manifests were found")
    return run_rows, checkpoint_rows, batch_rows, controller_rows, sources


def summarize(run_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in run_rows:
        key = (int(row["n"]), int(row["k"]), row["family"], row["method"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (n, k, family, method), group in sorted(grouped.items()):
        times = [float(r["external_wall_s"]) for r in group]
        sses = [float(r["final_exact_sse"]) for r in group]
        mt, q1t, q3t = quantiles(times)
        ms, q1s, q3s = quantiles(sses)
        summary.append({"n": n, "k": k, "family": family, "method": method,
            "run_count": len(group), "seeds": ",".join(sorted({r["seed"] for r in group}, key=int)),
            "median_wall_s": mt, "wall_iqr_s": q3t-q1t, "median_final_exact_sse": ms,
            "sse_iqr": q3s-q1s, "target_hits": sum(r["target_hit"].lower()=="true" for r in group),
            "target_hit_rate": statistics.mean(r["target_hit"].lower()=="true" for r in group),
            "all_runs_cuda": all(r.get("execution_device") == "cuda" for r in group)})

    cell: dict[tuple, list[dict]] = {}
    for row in run_rows:
        cell.setdefault((int(row["n"]), int(row["k"]), int(row["seed"]), row["family"]), []).append(row)
    comparisons = []
    for (n, k, seed, family), group in sorted(cell.items()):
        if family == "exact":
            continue
        adaptive_name = "QGCUDA" if family == "cumulative" else "ADAPTIVE_V2"
        prefix = "CUM_FIXED_" if family == "cumulative" else "DECAY_FIXED_"
        adaptive = next((r for r in group if r["method"] == adaptive_name), None)
        fixed = [r for r in group if r["method"].startswith(prefix)]
        exact = next((r for r in cell.get((n, k, seed, "exact"), []) if r["method"] == "G0"), None)
        if adaptive is None or not fixed:
            continue
        quality_best = min(fixed, key=lambda r: float(r["final_exact_sse"]))
        speed_best = min(fixed, key=lambda r: float(r["external_wall_s"]))
        comparisons.append({"n": n, "k": k, "seed": seed, "family": family,
            "adaptive_method": adaptive_name, "adaptive_wall_s": float(adaptive["external_wall_s"]),
            "adaptive_sse": float(adaptive["final_exact_sse"]), "adaptive_gap_percent": 100*float(adaptive["relative_gap_to_reference"]),
            "adaptive_target_hit": adaptive["target_hit"], "quality_best_fixed": quality_best["method"],
            "quality_best_fixed_wall_s": float(quality_best["external_wall_s"]),
            "quality_best_fixed_sse": float(quality_best["final_exact_sse"]),
            "adaptive_time_over_quality_best_fixed": float(adaptive["external_wall_s"])/float(quality_best["external_wall_s"]),
            "adaptive_sse_delta_percent_vs_quality_best_fixed": 100*(float(adaptive["final_exact_sse"])/float(quality_best["final_exact_sse"])-1),
            "fastest_fixed": speed_best["method"], "fastest_fixed_wall_s": float(speed_best["external_wall_s"]),
            "fastest_fixed_sse": float(speed_best["final_exact_sse"]),
            "adaptive_time_over_fastest_fixed": float(adaptive["external_wall_s"])/float(speed_best["external_wall_s"]),
            "adaptive_sse_delta_percent_vs_fastest_fixed": 100*(float(adaptive["final_exact_sse"])/float(speed_best["final_exact_sse"])-1),
            "any_fixed_target_hit": any(r["target_hit"].lower()=="true" for r in fixed),
            "time_comparison_at_frozen_target_valid": adaptive["target_hit"].lower()=="true" and any(r["target_hit"].lower()=="true" for r in fixed),
            "exact_g0_wall_s": float(exact["external_wall_s"]) if exact else "",
            "exact_g0_sse": float(exact["final_exact_sse"]) if exact else "",
            "exact_g0_target_hit": exact["target_hit"] if exact else ""})
    target = []
    for (n,k,family,method), group in sorted(grouped.items()):
        target.append({"n":n,"k":k,"family":family,"method":method,"runs":len(group),
            "seeds":",".join(sorted({r['seed'] for r in group},key=int)),
            "hits":sum(r['target_hit'].lower()=="true" for r in group),
            "target_hit_rate":statistics.mean(r['target_hit'].lower()=="true" for r in group),
            "median_time_to_quality_s":float(np.median([float(r['time_to_quality_s']) for r in group if r['time_to_quality_s']])) if any(r['time_to_quality_s'] for r in group) else ""})
    return summary, comparisons, target


def figures(run_rows: list[dict], checkpoints: list[dict], batch_rows: list[dict]) -> None:
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    for k in (8,16,32):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for method in sorted({r["method"] for r in run_rows if int(r["k"])==k}):
            group=[r for r in run_rows if int(r["k"])==k and r["method"]==method]
            byn={}
            for r in group: byn.setdefault(int(r["n"]),[]).append(r)
            xs=sorted(byn)
            axes[0].plot(xs,[np.median([float(x['external_wall_s']) for x in byn[n]]) for n in xs],marker='o',label=method)
            axes[1].plot(xs,[np.median([100*float(x['relative_gap_to_reference']) for x in byn[n]]) for n in xs],marker='o',label=method)
        axes[0].set(xscale='log',yscale='log',xlabel='Valid pixels N',ylabel='Complete-fit wall time (s)',title='All measured fits')
        axes[1].axhline(5,color='black',linestyle='--',linewidth=1,label='Frozen 5% target')
        axes[1].set(xscale='log',xlabel='Valid pixels N',ylabel='Final SSE gap to exact G0 (%)',title='Target misses retained')
        for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2)
        fig.suptitle(f'31UFS workload screening, K={k}; pilot plus selected seed replications')
        fig.tight_layout(); fig.savefig(figdir/f'workload_scalability_K{k}.png',dpi=180); plt.close(fig)
    # Adaptation schedules, de-duplicated from the raw source traces.
    keys=("method","family","n","k","seed","repeat","step","actual_batch_size","points_processed")
    unique={tuple(r.get(k,"") for k in keys):r for r in batch_rows}
    traces=list(unique.values())
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for method in ("QGCUDA","ADAPTIVE_V2"):
        group=[r for r in traces if r["method"]==method and int(r["k"])==16]
        byrun=_group(group,lambda r:(r["n"],r["seed"],r["repeat"]))
        for (n,seed,repeat),history in sorted(byrun.items(),key=lambda x:(int(x[0][0]),int(x[0][1]))):
            history=sorted({int(r["step"]):r for r in history if r.get("step")}.values(),key=lambda r:int(r["step"]))
            points=[int(r["points_processed"])/1e6 for r in history if r.get("points_processed")]
            batches=[int(r["actual_batch_size"]) for r in history if r.get("points_processed")]
            if points:
                axes[0 if method=="QGCUDA" else 1].step(points,batches,where="post",alpha=.8,label=f'N={int(n)/1e6:g}M, seed={seed}')
    for ax in axes: ax.set_ylabel('Batch size'); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02))
    axes[0].text(.01,.98,'QGCUDA (cumulative)',transform=axes[0].transAxes,va='top',fontsize=9,fontweight='bold',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none'})
    axes[1].text(.01,.98,'Adaptive V2 (decayed)',transform=axes[1].transAxes,va='top',fontsize=9,fontweight='bold',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none'})
    axes[1].set_xlabel('Processed observations (millions)')
    fig.suptitle('Adaptive batch-size schedules, K=16',y=.98)
    fig.subplots_adjust(top=.87,hspace=.55); fig.savefig(figdir/'adaptive_batch_schedules_K16.png',dpi=180); plt.close(fig)
    for k in (8,16,32):
        fig,ax=plt.subplots(figsize=(10,5.5))
        subset=[r for r in run_rows if int(r['k'])==k]
        methods=sorted({r['method'] for r in subset})
        for method in methods:
            group=[r for r in subset if r['method']==method]
            byn={}
            for r in group: byn.setdefault(int(r['n']),[]).append(r)
            xs=sorted(byn)
            ax.plot(xs,[statistics.mean(r['target_hit'].lower()=='true' for r in byn[n]) for n in xs],marker='o',label=method)
        ax.set(xscale='log',ylim=(-.05,1.05),xlabel='Valid pixels N',ylabel='Target hit fraction',title=f'Frozen 5% target attainment, K={k}')
        ax.grid(alpha=.25);ax.legend(fontsize=6,ncol=2)
        fig.tight_layout();fig.savefig(figdir/f'target_attainment_K{k}.png',dpi=180);plt.close(fig)
    for k in (8,16,32):
        fig,ax=plt.subplots(figsize=(9,5.2))
        subset=[r for r in run_rows if int(r['k'])==k]
        exact=[r for r in subset if r['method']=='G0' and r.get('time_to_quality_s')]
        if exact:
            byn=_group(exact,lambda r:int(r['n']))
            xs=sorted(byn)
            times=[float(np.median([float(r['time_to_quality_s']) for r in byn[n]])) for n in xs]
            ax.plot(xs,times,marker='o',linewidth=2,label='Exact CUDA G0 (target reached)')
            upper=max(times)*1.25
            for n in xs:
                ax.annotate('Mini-Batch: miss',xy=(n,upper),xytext=(0,2),textcoords='offset points',ha='center',fontsize=8)
            ax.margins(x=.2)
            ax.set(xscale='log',yscale='log',xlabel='Valid pixels N',ylabel='Observed time to frozen target (s)',title=f'Time-to-quality; Mini-Batch misses shown explicitly, K={k}',ylim=(min(times)*.75,upper*1.2))
            ax.legend()
        ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figdir/f'time_to_quality_K{k}.png',dpi=180);plt.close(fig)
    # Keep G0 full Lloyd-pass counts off the Mini-Batch observation axis. Each
    # K figure has one panel per N and separate cumulative/decayed method lines.
    for k in (8,16,32):
        fig,axes=plt.subplots(1,3,figsize=(16,5.5))
        methods=sorted({r['method'] for r in run_rows if int(r['k'])==k and r['family'] in ('cumulative','decayed')})
        handles=[]; labels=[]
        for ax,n in zip(axes,(262144,1048548,4194265)):
            subset=[r for r in checkpoints if int(r['k'])==k and int(r['n'])==n and r.get('points_processed') and r.get('family') in ('cumulative','decayed')]
            for method in methods:
                group=[r for r in subset if r['method']==method]
                if not group: continue
                bypoint=_group(group,lambda r:int(r['points_processed']))
                xs=sorted(bypoint)
                ys=[float(np.median([float(r['full_exact_sse']) for r in bypoint[x]])) for x in xs]
                line,=ax.plot([x/1e6 for x in xs],ys,marker='.',linewidth=1,label=method)
                if n==262144: handles.append(line); labels.append(method)
            targets=[float(r['target_sse']) for r in run_rows if int(r['n'])==n and int(r['k'])==k]
            if targets: ax.axhline(float(np.median(targets)),color='black',linestyle='--',linewidth=1)
            ax.set(title=f'N={n:,}',xlabel='Processed observations (millions)')
            ax.grid(alpha=.25)
        axes[0].set_ylabel('Exact full-data SSE at checkpoint')
        handles.append(plt.Line2D([],[],color='black',linestyle='--')); labels.append('Frozen 5% target')
        fig.suptitle(f'Mini-Batch quality at common observation checkpoints, K={k}')
        fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7,bbox_to_anchor=(.5,-.01))
        fig.subplots_adjust(bottom=.25,top=.84,wspace=.28)
        fig.savefig(figdir/f'sse_checkpoints_K{k}.png',dpi=180);plt.close(fig)


def _group(items, key):
    result={}
    for item in items: result.setdefault(key(item),[]).append(item)
    return result


def main() -> None:
    run_rows, checkpoints, batch_rows, controller_rows, sources = collect()
    summary, comparisons, target = summarize(run_rows)
    out=OUT/'final_summary'; out.mkdir(exist_ok=True)
    write_csv(out/'all_measured_runs.csv',run_rows)
    write_csv(out/'all_checkpoints.csv',checkpoints)
    timing_rows=[]
    for r in run_rows:
        try: parts=json.loads(r.get('timings') or '{}')
        except json.JSONDecodeError: parts={}
        timing_rows.append({"method":r["method"],"family":r["family"],"n":r["n"],"k":r["k"],
            "seed":r["seed"],"repeat":r["repeat"],"external_wall_s":r["external_wall_s"],
            **{f"{key}_s":value for key,value in parts.items()},
            "scope_note":"Internal categories may overlap or be subsets; external wall is primary and categories must not be summed."})
    write_csv(out/'all_timing_components.csv',timing_rows)
    write_csv(out/'method_summary_by_cell.csv',summary)
    write_csv(out/'adaptive_vs_fixed_by_seed.csv',comparisons)
    write_csv(out/'target_attainment_by_cell.csv',target)
    write_csv(out/'included_run_sources.csv',sources)
    write_csv(out/'controller_events.csv',controller_rows)
    figures(run_rows,checkpoints,batch_rows)
    total=len(run_rows)
    mini=[r for r in run_rows if r['family'] in ('cumulative','decayed')]
    g0=[r for r in run_rows if r['method']=='G0']
    mini_hits=sum(r['target_hit'].lower()=='true' for r in mini)
    g0_hits=sum(r['target_hit'].lower()=='true' for r in g0)
    text=f'''# Sentinel-2 workload scalability study — screening findings

## Scope and status

The study uses genuine Sentinel-2 Level-2A B02/B03/B04/B08 features from the 31UFS development acquisition, with development-fitted scaling. Crops contain N=262,144, 1,048,548 and 4,194,265 valid pixels, D=4; K=8,16,32. The crop sizes are nested spatial regions of one acquisition, not independent scenes. The frozen evaluator rule is 1.05 × a newly converged exact CUDA G0 SSE for the same N/K/seed. Controllers did not receive this target. Every Mini-Batch run was capped at 40.96M observations and stopped at the budget; none is described as converged.

The complete seed-0 pilot contains all nine N/K cells. Seeds 1 and 2 were added only for the promising 4M-pixel K=8 and K=16 cells. There is one measured fit per method per seed, not repeated timing trials per seed. Thus timing variability below is across seeds only in those selected cells, and the remainder are single-run screening results. Separate size-by-size seed-0 reruns are preserved and checksummed but excluded from the aggregate to avoid weighting seed 0 repeatedly.

## Results

- Complete measured fits: {total}; of these {len(g0)} are exact G0 and {len(mini)} are Mini-Batch. Exact G0 reached its per-cell 5% target in {g0_hits}/{len(g0)} fits. Mini-Batch reached it in {mini_hits}/{len(mini)} fits.
- At 262K and 1M, all fixed, QGCUDA and Adaptive V2 Mini-Batch fits missed the target for all tested K. Exact G0 reached the target. Therefore no Mini-Batch time-to-target comparison is valid at these sizes.
- At 4M, all Mini-Batch runs in the seed-0 sweep and the K=8/16 seed-1/2 replication missed the 5% target. Exact G0 reached it in every measured seed/cell. No Mini-Batch matched-quality speedup is established.
- At 4M, cumulative QGCUDA was faster than the lowest-SSE fixed cumulative batch in all three K=8 seeds; its final SSE differed from that fixed result by -0.037%, -0.011% and -0.001%. Its median complete time was 3.633 s versus 28.424 s for the lowest-SSE fixed width, but QGCUDA was not faster than the fastest fixed width in every seed (seed 0 had a 12.416 s QGCUDA run versus 6.968 s at fixed batch 65536). At K=16, QGCUDA was also faster than the lowest-SSE fixed batch in all three seeds, with per-seed SSE differences of -0.014%, +0.746% and +0.047%; it did not consistently beat the fastest fixed width. These are exploratory three-seed observations with one timing sample per seed and substantial between-seed/runtime variation.
- Adaptive V2 and fixed-decay results are a distinct update family. At 4M, Adaptive V2's K=8 SSE was within +0.007% to +0.023% of the lowest-SSE fixed-decay result and faster than that lowest-SSE fixed batch in all three seeds, but sometimes slower than the fastest large fixed batch. At K=16, its SSE was within -0.045% to +0.053% of the lowest-SSE fixed result and was faster than that quality-best fixed batch in all three seeds; it was also faster than the fastest fixed width in all three seeds in this one-sample-per-seed pilot. These indicate a possible time/quality tradeoff under equal observation budgets, not attainment of the frozen target or a universal win.
- Exact G0 was much faster than most Mini-Batch runs at 262K and 1M while reaching the target. At 4M, QGCUDA could finish sooner than G0 but failed the target, while G0 reached it. That is a fixed-budget speed/quality tradeoff, not time-to-quality superiority.

## Interpretation

Larger N did not make Mini-Batch reach the unchanged quality criterion under this 40.96M observation cap. The results support narrow equal-budget trade-offs: QGCUDA is much faster than the SSE-best small fixed cumulative batch with near-identical SSE at 4M, especially for K=8; Adaptive V2 at 4M/K=16 is faster and has lower SSE than the fastest fixed-decay width in all three observed seeds, while staying within about 0.05% of the SSE-best fixed-decay result. These are promising approximate-regime observations, but all Mini-Batch outputs remain outside the frozen 5% target. Thus the study does not establish an adaptive advantage at comparable target quality or time-to-target. Exact CUDA remains the supported choice when the 5% criterion is required.

## Limits

This is a single development acquisition and nested crops, not independent image scenes. The full N/K matrix has one seed and one measurement; only 4M K=8/16 has three seeds, still one measurement per seed. The timing values summarize different seeds, not repeated runs of a fixed seed; the wide timing spread limits confidence. G0 Lloyd-pass progress is not normalized to Mini-Batch observations. Exact quality is evaluated on the full current crop; no land-cover labels or external accuracy are available. Cumulative and decayed families use their respective probe/sampling policies and are not pooled. No GPU ANN is part of this study. Some per-run trajectory/provenance rows are duplicated by the runner; the original raw files are preserved and aggregate trajectory plots de-duplicate identical keys. Low-occupancy Numba warnings were emitted during warmup; CUDA device metadata and per-fit execution records confirm RTX 4050 CUDA execution.

## Reproduction

Use the prepared feature arrays already under `outputs/20261024-plan-real-prepared/`. The exact run commands and source/code hashes are in each complete child `config.json` and `manifest.json`; source hashes and included run directories are listed in `included_run_sources.csv`. The initial aborted attempt is retained separately and excluded; size-by-size seed-0 repeats are preserved but excluded from the pooled summary. Run `.venv/bin/python experiments/finalize_workload_scalability_study.py` to rebuild aggregate tables and figures from the complete primary pilot and selected confirmatory runs. Top-level checksums cover all files.
'''
    (out/'workload_scalability_findings.md').write_text(text,encoding='utf-8')
    (OUT/'README.md').write_text("""# 2026-10-31 workload scalability study\n\nThis is a reproducible, development-scene screening study. Real Sentinel-2 31UFS arrays are read from the preserved prepared-data directory; no satellite SAFE archive is duplicated here. See `final_summary/workload_scalability_findings.md` for results and caveats. Each complete run subdirectory contains its own config, manifest, raw CSVs, figures, hardware snapshot and SHA256 list. `pilot-seed0-corrected/` and `pilot-seed0/` are incomplete attempts retained with failure notes and excluded from aggregates.\n\nRebuild the aggregate tables and figures with `.venv/bin/python experiments/finalize_workload_scalability_study.py`. The study-run commands used `bash scripts/run_cuda_python.sh experiments/workload_scalability_study.py`; full arguments and code hashes are recorded in each run's `config.json`. Do not combine cumulative and decayed update families.\n""",encoding='utf-8')
    supplementary=[]
    for manifest_path in sorted((OUT/'pilot-seed0').glob('N*/manifest.json')):
        m=json.loads(manifest_path.read_text())
        if m.get('complete'):
            supplementary.append({"run_directory":str(manifest_path.parent.relative_to(ROOT)),
                "manifest_sha256":sha(manifest_path),"completed_measured_fits":m.get('completed_measured_fits'),
                "status":"complete; excluded from primary aggregate to avoid repeated weighting of seed 0"})
    root_manifest={"study":"Sentinel-2 workload scalability study","complete_pilot_matrix":True,
        "included_complete_runs":sources,"excluded_incomplete_attempts":[
            "pilot-seed0/ (operator-aborted initial schema mismatch; retained separately)"],
        "supplementary_complete_runs_excluded_from_aggregate":supplementary,
        "measured_fit_count":total,"mini_target_hits":mini_hits,"mini_fit_count":len(mini),
        "exact_g0_target_hits":g0_hits,"exact_g0_fit_count":len(g0),
        "finalizer_sha256":sha(Path(__file__)),"analysis_note":"No target-matched Mini-Batch result; no Mini-Batch time-to-quality claim.",
        "manifest_created_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    (OUT/'manifest.json').write_text(json.dumps(root_manifest,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('checksums.sha256','experiment_index.json'))
    index={"root":str(OUT.resolve()),"manifest":"manifest.json","files":[
        {"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in files],
        "checksums_file":"checksums.sha256"}
    (OUT/'experiment_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (OUT/'checksums.sha256').write_text(''.join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files),encoding='utf-8')
    print(f"Aggregated {total} complete measured fits; mini target hits={mini_hits}/{len(mini)}; G0 hits={g0_hits}/{len(g0)}")


if __name__ == '__main__':
    main()
