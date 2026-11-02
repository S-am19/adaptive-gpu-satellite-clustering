"""Compare quality-guarded adaptation with fixed CUDA batch-size baselines.

The experiment reuses the experiment-only cumulative update orchestration and
the already validated Phase 3 policies. It requires real CUDA and refuses the
CUDA simulator. All methods use a common disjoint fixed probe and an exact
40.96M processed-observation budget; target/reference values are evaluator-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pcap.adaptive_controller import BatchPolicy
from pcap.contracts import FitConfig
from pcap.controlled_fit import fit_controlled
from pcap.initialization import initialize_centroids
from pcap.quality_guard_controller import QualityGuardPolicy
from pcap.budgeted_minibatch import fit_budgeted_minibatch
from run_suite import provenance, snapshot_sources, validate_result


METHODS = ["FIXED_8192", "FIXED_16384", "FIXED_32768", "FIXED_65536",
           "QGCUDA", "G3CUDA", "G0"]
FIXED_BATCHES = {"FIXED_8192": 8192, "FIXED_16384": 16384,
                 "FIXED_32768": 32768, "FIXED_65536": 65536}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def gpu_conditions() -> str:
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.free,memory.total,pstate,temperature.gpu,clocks.sm,utilization.gpu",
            "--format=csv"], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def aggregate(runs: list[dict]) -> list[dict]:
    result = []
    for method in METHODS:
        subset = [r for r in runs if r["method"] == method]
        if not subset:
            continue
        walls = np.asarray([float(r["external_wall_s"]) for r in subset])
        sses = np.asarray([float(r["final_exact_sse"]) for r in subset])
        targets = [r["target_hit"] == "True" for r in subset]
        crossings = [float(r["first_target_crossing_time_s"]) for r in subset
                     if r.get("first_target_crossing_time_s") not in ("", None)]
        result.append(dict(
            method=method, execution_device=subset[0]["device"], n_runs=len(subset),
            median_external_wall_s=float(np.median(walls)),
            wall_iqr_s=float(np.percentile(walls, 75) - np.percentile(walls, 25)),
            median_final_exact_sse=float(np.median(sses)),
            final_sse_iqr=float(np.percentile(sses, 75) - np.percentile(sses, 25)),
            median_sse_gap_to_exact_reference=float(np.median(
                [float(r["sse_gap_to_exact_reference"]) for r in subset])),
            target_hits=sum(targets), target_hit_rate=float(np.mean(targets)),
            median_processed_observations=float(np.median(
                [int(r["processed_observations"]) for r in subset])),
            median_committed_observations=float(np.median(
                [int(r["committed_observations"]) for r in subset])),
            median_steps=float(np.median([int(r["steps"]) for r in subset])),
            median_first_target_crossing_s=float(np.median(crossings)) if crossings else "",
            n_target_crossings=len(crossings),
            termination_reasons=";".join(sorted({r["termination_reason"] for r in subset})),
            mean_monitoring_s=float(np.mean([float(r["timings"].get("monitoring_s", 0.0)) for r in subset])),
            mean_controller_s=float(np.mean([float(r["timings"].get("controller_s", 0.0)) for r in subset])),
            mean_quality_evaluation_s=float(np.mean([float(r["timings"].get("quality_evaluation_s", 0.0)) for r in subset])),
        ))
    return result


def make_plots(out: Path, runs: list[dict], checkpoints: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    summaries = aggregate(runs)
    labels = [r["method"] for r in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, method in enumerate(labels):
        rr = [r for r in runs if r["method"] == method]
        axes[0].scatter([i] * len(rr), [float(r["external_wall_s"]) for r in rr], alpha=.7)
        axes[1].scatter([i] * len(rr), [float(r["final_exact_sse"]) for r in rr], alpha=.7)
    axes[0].set_ylabel("Complete fit wall time (s)")
    axes[1].set_ylabel("Final exact full-data SSE")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Fixed batch-size ablation vs adaptive CUDA controllers; individual seed/repeats")
    fig.tight_layout()
    fig.savefig(figdir / "runtime_and_final_sse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for color, method in zip(colors, labels):
        method_rows = [r for r in checkpoints if r["method"] == method]
        grouped = {}
        for row in method_rows:
            grouped.setdefault(int(row["checkpoint_index"]), []).append(float(row["full_exact_sse"]))
        xx = sorted(grouped)
        yy = [float(np.median(grouped[j])) for j in xx]
        obs = [statistics.median(int(r["processed_observations"]) for r in method_rows
                                 if int(r["checkpoint_index"]) == j) for j in xx]
        if xx:
            ax.plot(np.asarray(obs) / 1e6, yy, marker="o", label=method, color=color)
    ax.set(xlabel="Processed observations (millions)", ylabel="Exact full-data SSE",
           title="Common quality checkpoints; target is evaluation-only")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(figdir / "sse_vs_processed_observations.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, method in enumerate(labels):
        rr = [r for r in runs if r["method"] == method]
        hits = sum(r["target_hit"] == "True" for r in rr)
        axes[0].bar(i, hits / len(rr), color=colors[i])
        crossing = [float(r["first_target_crossing_time_s"]) for r in rr
                    if r.get("first_target_crossing_time_s") not in ("", None)]
        if crossing:
            axes[1].scatter([i] * len(crossing), crossing, color=colors[i])
    axes[0].set_ylabel("Target hit rate")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("First observed target crossing (s; hits only)")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Frozen 5% quality target: evaluation only")
    fig.tight_layout(); fig.savefig(figdir / "time_to_quality_and_hit_rate.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in labels:
        r = next((x for x in runs if x["method"] == method and int(x["seed"]) == 0 and int(x["repeat"]) == 0), None)
        if not r or not r["batch_schedule"]:
            continue
        sched = (r["batch_schedule"] if isinstance(r["batch_schedule"], list)
                 else json.loads(r["batch_schedule"]))
        ax.step([x["points_processed"] / 1e6 for x in sched],
                [x["actual_batch_size"] for x in sched], where="post", label=method)
    ax.set(xlabel="Processed observations (millions)", ylabel="Actual batch size",
           title="Batch trajectories (seed 0, repeat 0)")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(figdir / "batch_size_trajectories.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels)); width = .25
    for offset, key, label in ((-width, "monitoring_s", "probe monitoring"),
                               (0, "controller_s", "controller"),
                               (width, "quality_evaluation_s", "full-data evaluation")):
        values = [float(s["mean_" + key]) for s in summaries]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Mean internal timing (s per fit)")
    ax.set_title("Monitoring and control overhead (categories are not additive to wall time)")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figdir / "monitoring_controller_overhead.png", dpi=180); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "outputs/20261025-independent-holdouts/validation512")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/20261027-adaptive-four-phase-study/phase3_quality_guarded_adaptive/full_disjoint_probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--observation-budget", type=int, default=40_960_000)
    parser.add_argument("--evaluation-interval", type=int, default=8_192_000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--time-budget-seconds", type=float, default=4200)
    args = parser.parse_args(argv)
    scene_dir, ref_dir, out = args.scene_dir.resolve(), args.reference_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Real CUDA GPU required; simulator and CPU fallback are disabled")
    X = np.load(scene_dir / "X.npy", allow_pickle=False)
    scene = json.loads((scene_dir / "metadata.json").read_text())
    if X.shape != (262144, 4) or X.dtype != np.float32 or not X.flags.c_contiguous or not np.isfinite(X).all():
        raise ValueError(f"Expected the prepared finite float32 validation crop [262144,4], got {X.shape}/{X.dtype}")
    x_hash = sha(X.tobytes())
    if scene.get("X_sha256") != x_hash:
        raise ValueError("Validation feature hash does not match scene metadata")
    phase3_config = json.loads((ref_dir / "config.json").read_text())
    phase3_refs = list(csv.DictReader((ref_dir / "reference_runs.csv").open(newline="")))
    refs = {int(r["seed"]): r for r in phase3_refs}
    if any(seed not in refs for seed in args.seeds):
        raise ValueError("Each requested seed needs a verified Phase 3 exact reference")
    if args.repeats < 1 or args.warmups < 0 or args.observation_budget % args.evaluation_interval:
        raise ValueError("Invalid repeats/warmups or budget/checkpoint divisibility")
    if args.evaluation_interval % 65536:
        raise ValueError("For fair fixed-batch checkpoints, evaluation interval must be divisible by 65536")

    batch_policy = BatchPolicy(**phase3_config["batch_policy"])
    qg_policy = QualityGuardPolicy(**phase3_config["quality_guard_policy"])
    config = dict(
        experiment="fixed_batch_size_vs_existing_adaptive_cuda_controllers",
        dataset=scene, scene_id=scene.get("scene_id"), n=int(len(X)), d=int(X.shape[1]), k=args.k,
        seeds=args.seeds, measured_repetitions=args.repeats, warmups=args.warmups,
        methods=METHODS, fixed_batch_sizes=FIXED_BATCHES,
        adaptive_batch_policy=batch_policy.__dict__, quality_guard_policy=qg_policy.__dict__,
        exact_reference="G0 exact CUDA Lloyd; seed-matched C0 and evaluation-only target from Phase 3",
        observation_budget=args.observation_budget, evaluation_interval=args.evaluation_interval,
        max_steps=args.max_steps, tolerance=1e-6, probe_interval=10, probe_size=1024,
        disjoint_probe=True, probe_seed="SeedSequence([seed,1])",
        frozen_target="Phase3 exact CPU reference SSE * 1.05; passed only to exact G0 evaluator/reporting; never to QG/G3 controllers",
        sampling_policy="Each run uses numpy.default_rng(seed), uniform without replacement per batch over the complement of the identical fixed 1024-point probe. Batch-size-specific calls consume different index streams; no pointwise-identical sample sequence is claimed.",
        update_rule="Cumulative float64 sums and int64 counts; empty clusters retain centroids; exact squared-Euclidean batch assignment",
        checkpoint_policy="Exact full-data SSE after each common 8.192M observation checkpoint and at final output; fixed probe at probe_interval and checkpoints",
        warmup_policy="Two excluded warm-up fits per method at seed 0; fixed arms use their configured batch width, adaptive arms use two initial-size batches; G0 warm-up exact target-stopped fit",
        measurement="External perf_counter over complete fit call; internal timing fields separately reported and overlapping fields not summed",
        execution_order="Randomized method order independently within each seed/repetition by seeded scheduler; schedule saved in individual_runs.csv",
        x_sha256=x_hash, cuda_simulator=False, cuda_device=str(cuda.get_current_device().name),
        source_reference_dir=str(ref_dir), prior_phase3_config_sha256=sha((ref_dir / "config.json").read_bytes()),
        git_commit=provenance()["git_commit"],
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    source_info = provenance()
    manifest = dict(source_info, created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                    failures=[], config=config, source_hashes=source_info["source_hashes"],
                    gpu_conditions_before=gpu_conditions(), ram_before=subprocess.check_output(["free", "-b"], text=True),
                    protocol_note="New focused follow-up. Original G1/G3CUDA/QGCUDA implementations and previous archives were not changed.")
    snapshot_sources(out / "source", manifest["source_hashes"])

    individual, checkpoints, events, warmups, references = [], [], [], [], []
    order_rng = np.random.default_rng(20261027)
    start_all = perf_counter()
    try:
        # Warm up each exact batch-width path and both adaptive controllers.
        seed0 = args.seeds[0]
        C0 = initialize_centroids(X, args.k, seed=seed0)
        ref0 = refs[seed0]
        target0 = float(ref0["target_sse"])
        for method in order_rng.permutation(METHODS):
            for warm in range(args.warmups):
                if method == "G0":
                    cfg = FitConfig(args.k, seed0, args.max_steps, 1e-6)
                    t = perf_counter()
                    fit_controlled(X, C0, cfg, device="cuda", mode="full",
                                   evaluation_interval=1, target_sse=target0)
                else:
                    b = FIXED_BATCHES.get(method, 8192)
                    cfg = FitConfig(args.k, seed0, args.max_steps, 1e-6, b, 10)
                    budget = b if method.startswith("FIXED_") else 2 * b
                    kwargs = dict(device="cuda", observation_budget=budget,
                                  evaluation_interval=budget, disjoint_probe=True)
                    if method == "G3CUDA":
                        kwargs.update(adaptive=True, batch_policy=batch_policy)
                    elif method == "QGCUDA":
                        kwargs.update(quality_guard_policy=qg_policy)
                    t = perf_counter()
                    fit_budgeted_minibatch(X, C0, cfg, **kwargs)
                warmups.append(dict(method=method, seed=seed0, warmup=warm,
                                    wall_s=perf_counter()-t, excluded=True,
                                    warmed_batch=FIXED_BATCHES.get(method, 8192)))
        write_csv(out / "warmups.csv", warmups)

        for seed in args.seeds:
            C0 = initialize_centroids(X, args.k, seed=seed)
            c_hash = sha(C0.tobytes())
            ref = refs[seed]
            if ref["C0_sha256"] != c_hash:
                raise ValueError(f"C0 differs from Phase 3 reference for seed {seed}")
            ref_sse, target = float(ref["reference_sse"]), float(ref["target_sse"])
            references.append(dict(seed=seed, x_sha256=x_hash, C0_sha256=c_hash,
                                   exact_cpu_reference_sse=ref_sse, frozen_target_sse=target,
                                   target_is_evaluation_only=True))
            for repeat in range(args.repeats):
                schedule = [METHODS[i] for i in order_rng.permutation(len(METHODS))]
                for order_position, method in enumerate(schedule):
                    if perf_counter() - start_all > args.time_budget_seconds:
                        raise TimeoutError("time budget checked between complete method fits")
                    b = FIXED_BATCHES.get(method, 8192)
                    cfg = FitConfig(args.k, seed, args.max_steps, 1e-6,
                                    None if method == "G0" else b, 10)
                    t = perf_counter()
                    if method == "G0":
                        result = fit_controlled(X, C0, cfg, device="cuda", mode="full",
                                                evaluation_interval=1, target_sse=target)
                        cps = [dict(step=int(q["step"]), processed_observations=int(q["step"] * len(X)),
                                    full_exact_sse=float(q["full_data_sse"]), elapsed_s=float(q["elapsed_s"]),
                                    checkpoint_kind="each_exact_lloyd_iteration")
                               for q in result.metadata.get("quality_history", [])]
                        event_rows = []
                    else:
                        kwargs = dict(device="cuda", observation_budget=args.observation_budget,
                                      evaluation_interval=args.evaluation_interval, disjoint_probe=True)
                        if method == "G3CUDA":
                            kwargs.update(adaptive=True, batch_policy=batch_policy)
                        elif method == "QGCUDA":
                            kwargs.update(quality_guard_policy=qg_policy)
                        result, cps, event_rows = fit_budgeted_minibatch(X, C0, cfg, **kwargs)
                    external_wall = perf_counter() - t
                    validate_result(X, result)
                    if method == "G0":
                        quality_history = result.metadata.get("quality_history", [])
                        first = next((q for q in quality_history
                                      if float(q.get("full_data_sse", float("inf"))) <= target), None)
                        hit = bool(result.sse <= target)
                        first_obs = int(first["step"] * len(X)) if first else None
                        first_time = float(first["elapsed_s"]) if first else None
                        processed = int(result.metadata.get("points_processed", 0))
                        committed = processed
                    else:
                        first = next((q for q in cps if float(q["full_exact_sse"]) <= target), None)
                        hit = bool(first is not None or result.sse <= target)
                        first_obs = int(first["points_processed"]) if first else None
                        first_time = float(first["elapsed_s"]) if first else None
                        processed = int(result.metadata["points_processed"])
                        committed = int(result.metadata.get("committed_observations", processed))
                    row = dict(method=method, device="cuda", gpu=str(cuda.get_current_device().name),
                        seed=seed, repeat=repeat, order_position=order_position, randomized_order=schedule,
                        n=len(X), d=X.shape[1], k=args.k, X_sha256=x_hash, C0_sha256=c_hash,
                        batch_size=FIXED_BATCHES.get(method, "adaptive" if method != "G0" else "full Lloyd"),
                        target_sse=target, target_is_evaluation_only=True,
                        final_exact_sse=result.sse,
                        sse_gap_to_exact_reference=(result.sse-ref_sse)/ref_sse,
                        target_hit=hit, first_target_crossing_observations=first_obs,
                        first_target_crossing_time_s=first_time,
                        external_wall_s=external_wall, processed_observations=processed,
                        committed_observations=committed,
                        rolled_back_observations=int(result.metadata.get("observations_rolled_back", 0)),
                        steps=int(result.metadata.get("steps", 0)),
                        termination_reason=result.metadata.get("termination_reason"),
                        convergence_satisfied=bool(result.metadata.get("convergence_satisfied", False)),
                        batch_schedule=result.history if method != "G0" else [],
                        sampling_policy=result.metadata.get("sampling_policy", "full Lloyd; no mini-batch sampling"),
                        sampling_rng=result.metadata.get("sampling_rng", "not applicable"),
                        probe_size=result.metadata.get("probe_size", 0),
                        disjoint_probe=result.metadata.get("probe_is_disjoint", False),
                        timings=result.timings, metadata=result.metadata)
                    individual.append(row)
                    for checkpoint_index, q in enumerate(cps, start=1):
                        checkpoints.append(dict(method=method, seed=seed, repeat=repeat,
                            checkpoint_index=checkpoint_index,
                            processed_observations=q.get("points_processed", q.get("processed_observations")),
                            full_exact_sse=q.get("full_exact_sse"), elapsed_s=q.get("elapsed_s"),
                            checkpoint_kind=q.get("checkpoint_kind", "budget")))
                    for e in result.history:
                        if e.get("controller_decision") is not None or e.get("rollback"):
                            events.append(dict(method=method, seed=seed, repeat=repeat, **e))
                    for e in event_rows:
                        events.append(dict(method=method, seed=seed, repeat=repeat, **e))
                    write_csv(out / "individual_runs.csv", individual)
                    write_csv(out / "checkpoints.csv", checkpoints)
                    write_csv(out / "controller_events.csv", events)
                    write_csv(out / "reference_runs.csv", references)
                    write_csv(out / "timing_breakdown.csv", [dict(method=r["method"], seed=r["seed"],
                        repeat=r["repeat"], external_wall_s=r["external_wall_s"],
                        internal_wall_s=r["timings"].get("wall_s"),
                        **{k: v for k, v in r["timings"].items() if k != "wall_s"},
                        timing_note="categories can overlap; external wall is primary") for r in individual])
        summary = aggregate(individual)
        write_csv(out / "method_summary.csv", summary)
        make_plots(out, individual, checkpoints)
        report = [
            "# QGCUDA versus fixed batch-size follow-up",
            "",
            f"Scene: {scene.get('scene_id')}; N={len(X)}, D={X.shape[1]}, K={args.k}; seeds={args.seeds}; repeats={args.repeats}.",
            "All Mini-Batch methods use cumulative statistics, exact assignment, the same 1,024-point disjoint probe, common SSE checkpoints, and the same processed-observation budget. Batch-size-specific RNG calls produce different index sequences; the policy and seeds are identical, not pointwise sample lists.",
            "The 5% target is evaluator-only. G0 is exact CUDA Lloyd and stops at target/convergence; its full-data assignment work is not a 40.96M Mini-Batch budget.",
            "",
            "| Method | Median wall (s) | IQR (s) | Median exact SSE | Target hits | Median steps |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in summary:
            report.append(f"| {s['method']} | {s['median_external_wall_s']:.3f} | {s['wall_iqr_s']:.3f} | {s['median_final_exact_sse']:.2f} | {s['target_hits']}/{s['n_runs']} | {s['median_steps']:.0f} |")
        best_fixed = min((s for s in summary if s["method"].startswith("FIXED_")),
                         key=lambda s: s["median_external_wall_s"])
        qg = next(s for s in summary if s["method"] == "QGCUDA")
        report += ["", f"The best fixed batch by median runtime was {best_fixed['method']} at {best_fixed['median_external_wall_s']:.3f}s. QGCUDA median runtime was {qg['median_external_wall_s']:.3f}s. The paired seed/repetition rows, not only these medians, are preserved in `individual_runs.csv`.",
                   "", "No claim of time-to-quality improvement is valid for a method with zero target hits. Interpret fixed versus adaptive differences only with the matched exact SSE, target-hit, observation, update-count, and overhead columns."]
        (out / "final_summary.md").write_text("\n".join(report) + "\n")
        manifest["complete"] = True
    except Exception as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(elapsed_wall_s=perf_counter()-start_all,
                        completed_measured_fits=len(individual),
                        gpu_conditions_after=gpu_conditions(),
                        ram_after=subprocess.check_output(["free", "-b"], text=True))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
        (out / "checksums.sha256").write_text("".join(
            f"{sha(p.read_bytes())}  {p.relative_to(out)}\n" for p in sorted(files)))
    print(f"Completed {len(individual)} measured fits in {out}")


if __name__ == "__main__":
    main()
