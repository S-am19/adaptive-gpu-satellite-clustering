"""Replicate fixed-decay vs Adaptive V2 on the existing 31UFU validation crop."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numba import cuda
from pcap.adaptive_v2_controller import AdaptiveV2Policy
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.minibatch_decayed_cuda import (
    fit_adaptive_decayed_minibatch_cuda, fit_decayed_minibatch_cuda,
)
from run_suite import provenance, snapshot_sources, validate_result


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, records):
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        # Checkpoint/event rows can carry additional fields at terminal states.
        # Union their schemas so those rows are preserved instead of aborting.
        fields = list(dict.fromkeys(key for row in records for key in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def summarize(records):
    result = []
    for method in sorted({r["method"] for r in records}):
        subset = [r for r in records if r["method"] == method]
        walls = [float(r["wall_s"]) for r in subset]
        sses = [float(r["final_exact_sse"]) for r in subset]
        result.append(dict(method=method, n=len(subset), median_wall_s=statistics.median(walls),
            wall_iqr_s=float(np.percentile(walls, 75)-np.percentile(walls, 25)),
            median_final_exact_sse=statistics.median(sses),
            final_sse_iqr=float(np.percentile(sses, 75)-np.percentile(sses, 25)),
            target_hits=sum(bool(r["target_hit"]) for r in subset),
            target_hit_rate=sum(bool(r["target_hit"]) for r in subset)/len(subset),
            median_observations=statistics.median(int(r["observations_processed"]) for r in subset),
            median_steps=statistics.median(int(r["steps"]) for r in subset),
            termination_reasons=";".join(sorted({r["termination_reason"] for r in subset}))))
    return result


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("real CUDA required; simulator/CPU fallback is disabled")
    X = np.load(args.scene_dir / "X.npy", allow_pickle=False)
    scene = json.loads((args.scene_dir / "metadata.json").read_text())
    if X.dtype != np.float32 or X.shape != (262144, 4) or not X.flags.c_contiguous or not np.isfinite(X).all():
        raise ValueError("expected the prepared finite validation crop [262144,4]")
    x_hash = digest(X.tobytes())
    if scene.get("X_sha256") != x_hash:
        raise ValueError("validation feature hash mismatch")
    phase3_refs = list(csv.DictReader((args.reference_dir / "reference_runs.csv").open()))
    refs = {int(r["seed"]): r for r in phase3_refs}
    checkpoints = list(range(args.checkpoint_interval, args.observation_budget + 1,
                             args.checkpoint_interval))
    if not checkpoints or checkpoints[-1] != args.observation_budget:
        raise ValueError("observation budget must be divisible by checkpoint interval")
    policy = AdaptiveV2Policy.for_run(len(X), args.batch_size,
        minimum=args.minimum_batch, maximum=args.maximum_batch,
        deterioration=.01, stagnation_improvement=.001, progress_improvement=.005,
        quiet_movement=.001, quiet_label_change=.005, required_checkpoints=3)
    config = dict(scene=scene, n=len(X), d=X.shape[1], k=args.k,
        seeds=args.seeds, measured_repetitions=1, warmups=args.warmups,
        initial_batch=args.batch_size, minimum_batch=args.minimum_batch,
        maximum_batch=args.maximum_batch, max_steps=args.max_steps,
        observation_budget=args.observation_budget,
        quality_observation_checkpoints=checkpoints,
        probe_interval=args.probe_interval, quality_interval=0,
        decay_rule="lambda=0.5**(actual_batch/(10*N)); S=lambda*S+batch_sum; W=lambda*W+batch_count",
        decay_half_life_passes=10, quality_gap=.05,
        target_visibility="evaluation only; not passed to either fit/controller",
        update_family="decayed historical sums/effective counts; distinct from cumulative G1/G3/QG",
        sampling="seeded uniform without replacement within each batch; same seed policy, streams differ when batch schedules differ",
        x_sha256=x_hash, methods=["C_DECAYED_FIXED", "D_ADAPTIVE_V2"],
        policy=policy.__dict__, reference_source=str(args.reference_dir),
        source_scene_id=scene.get("scene_id"))
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    manifest = dict(provenance(), created_utc=datetime.now(timezone.utc).isoformat(),
        complete=False, failures=[], config=config, source_hashes=provenance()["source_hashes"],
        study_scope="separate replication of the decayed-count fixed vs Adaptive V2 update family")
    snapshot_sources(out / "source", manifest["source_hashes"])
    runs, checkpoint_rows, event_rows, warmup_rows, reference_rows = [], [], [], [], []
    started = perf_counter()
    try:
        # One short warm-up per implementation; excluded from all measured fields.
        warm_budget = min(2 * args.batch_size, args.observation_budget)
        for seed in args.seeds[:1]:
            C0 = initialize_centroids(X, args.k, seed=seed)
            cfg = FitConfig(args.k, seed, args.max_steps, 1e-6, args.batch_size, args.probe_interval)
            for method in config["methods"]:
                start = perf_counter()
                if method == "C_DECAYED_FIXED":
                    fit_decayed_minibatch_cuda(X, C0, cfg, quality_interval=0,
                        observation_budget=warm_budget)
                else:
                    fit_adaptive_decayed_minibatch_cuda(X, C0, cfg, policy=policy,
                        quality_interval=0, observation_budget=warm_budget)
                warmup_rows.append(dict(method=method, seed=seed, wall_s=perf_counter()-start, excluded=True))
        write_csv(out / "warmups.csv", warmup_rows)
        order_rng = np.random.default_rng(20261027)
        for seed in args.seeds:
            if seed not in refs:
                raise ValueError(f"Missing verified exact reference for seed {seed}")
            ref = refs[seed]
            C0 = initialize_centroids(X, args.k, seed=seed)
            c_hash = digest(C0.tobytes())
            if c_hash != ref["C0_sha256"]:
                raise ValueError(f"C0 hash differs from Phase 3 exact reference for seed {seed}")
            target = float(ref["target_sse"])
            reference_rows.append(dict(seed=seed, C0_sha256=c_hash,
                exact_cpu_sse=float(ref["reference_sse"]), target_sse=target,
                target_evaluation_only=True))
            cfg = FitConfig(args.k, seed, args.max_steps, 1e-6, args.batch_size, args.probe_interval)
            methods = list(config["methods"])
            if order_rng.integers(2): methods.reverse()
            for method in methods:
                t = perf_counter()
                if method == "C_DECAYED_FIXED":
                    result = fit_decayed_minibatch_cuda(X, C0, cfg, quality_interval=0,
                        observation_budget=args.observation_budget,
                        quality_observation_checkpoints=checkpoints)
                else:
                    result = fit_adaptive_decayed_minibatch_cuda(X, C0, cfg, policy=policy,
                        quality_interval=0, observation_budget=args.observation_budget,
                        quality_observation_checkpoints=checkpoints)
                wall = perf_counter()-t
                validate_result(X, result)
                hist = result.metadata.get("quality_history", [])
                first = next((q for q in hist if float(q["full_data_sse"]) <= target), None)
                run = dict(method=method, device="cuda", seed=seed, repeat=0,
                    n=len(X), d=X.shape[1], k=args.k, x_sha256=x_hash, C0_sha256=c_hash,
                    target_sse=target, target_evaluation_only=True,
                    final_exact_sse=result.sse,
                    gap_to_reference=(result.sse-float(ref["reference_sse"]))/float(ref["reference_sse"]),
                    target_hit=bool(first is not None or result.sse <= target),
                    first_target_crossing_observations=(int(first["points_processed"]) if first else None),
                    first_target_crossing_time_s=(float(first["elapsed_s"]) if first else None),
                    wall_s=wall, observations_processed=result.metadata["points_processed"],
                    # The decayed-count API records processed points and effective
                    # cluster weights, but has no separate committed-observation
                    # counter. Keep the schema explicit rather than infer it.
                    committed_observations=result.metadata.get("committed_observations"),
                    steps=result.metadata["steps"], termination_reason=result.metadata["termination_reason"],
                    convergence_satisfied=result.metadata["convergence_satisfied"],
                    batch_transitions=result.metadata.get("batch_transitions"),
                    timings=result.timings, metadata=result.metadata,
                    batch_schedule=[(h.get("points_processed"),h.get("actual_batch_size"),
                                     h.get("next_batch_size")) for h in result.history])
                runs.append(run)
                np.savez_compressed(out / f"arrays_{method}_seed{seed}.npz",
                    centroids=result.centroids, labels=result.labels,
                    final_exact_sse=np.asarray(result.sse), C0=C0)
                for item in hist:
                    checkpoint_rows.append(dict(method=method,seed=seed,repeat=0,**item))
                for item in result.history:
                    if item.get("controller_decision") is not None:
                        event_rows.append(dict(method=method,seed=seed,repeat=0,**item))
                with (out / "updates.jsonl").open("a") as f:
                    for item in result.history:
                        f.write(json.dumps(dict(method=method,seed=seed,repeat=0,**item), default=str)+"\n")
                write_csv(out / "individual_runs.csv", runs)
                write_csv(out / "checkpoints.csv", checkpoint_rows)
                write_csv(out / "controller_events.csv", event_rows)
                write_csv(out / "reference_runs.csv", reference_rows)
                write_csv(out / "timing_breakdown.csv", [dict(method=r["method"],seed=r["seed"],
                    repeat=r["repeat"],external_wall_s=r["wall_s"],
                    internal_wall_s=r["timings"].get("wall_s"),
                    **{k:v for k,v in r["timings"].items() if k != "wall_s"},
                    timing_scope="internal categories overlap; do not sum") for r in runs])
        write_csv(out / "method_summary.csv", summarize(runs))
        manifest["complete"] = True
    except Exception as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_wall_s"] = perf_counter()-started
        manifest["completed_measured_fits"] = len(runs)
        manifest["references"] = reference_rows
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
        (out / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n"
            for p in sorted(files)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--minimum-batch", type=int, default=256)
    p.add_argument("--maximum-batch", type=int, default=65536)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=10)
    p.add_argument("--observation-budget", type=int, default=40960000)
    p.add_argument("--checkpoint-interval", type=int, default=8192000)
    p.add_argument("--warmups", type=int, default=1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
