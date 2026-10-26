"""Reproducible pilot runner. G0/G1 require CUDA; G2-G5 are CPU prototypes.

No fallback is allowed. Raw results retain fit history, metadata and timings.
This runner does not establish time-to-target quality or final scene evaluation.
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import shutil
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcap.contracts import FitConfig, validate_features
from pcap.initialization import initialize_centroids
from pcap.metrics import evaluate_clustering_quality

METHODS = {"O0": "cpp_openmp_lloyd", "C1": "cpu_minibatch_exact", "C0": "cpu_exact", "G0": "cuda_exact", "G1": "cuda_minibatch",
           "G2": "cpu_minibatch_ann", "G3": "cpu_adaptive_batch",
           "G4": "cpu_adaptive_ann", "G5": "cpu_adaptive_combined",
           "G3CUDA": "cuda_adaptive_batch", "G2CUDA": "cuda_projection_ann_fixed",
           "G4CUDA": "cuda_projection_ann_adaptive_effort", "G5CUDA": "cuda_projection_ann_joint"}


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_synthetic_dataset(n, d, k, seed):
    rng = np.random.default_rng(seed)
    anchors = rng.normal(0, 3, (k, d))
    points = np.ascontiguousarray(anchors[rng.integers(0, k, n)] + rng.normal(0, .4, (n, d)), dtype=np.float32)
    return points, initialize_centroids(points, k, seed=seed)


def run_single_method(method, points, initial, cfg, *, policy_settings=None, omp_threads=1):
    if method == "O0":
        from pcap.kmeans_openmp import fit_openmp
        return fit_openmp(points,initial,replace(cfg,batch_size=None,ann_effort=None),threads=omp_threads)
    settings=policy_settings or {}
    batch_policy=ann_policy=None
    if settings.get("batch") and method in ("G3","G5","G3CUDA","G5CUDA"):
        from pcap.adaptive_controller import BatchPolicy
        batch_policy=BatchPolicy.for_run(len(points),cfg.batch_size,**settings["batch"])
    if settings.get("ann_cuda" if method.endswith("CUDA") else "ann_cpu") and method in ("G4","G5","G4CUDA","G5CUDA"):
        from pcap.adaptive_ann_controller import ANNEffortPolicy
        ann_policy=ANNEffortPolicy(**settings["ann_cuda" if method.endswith("CUDA") else "ann_cpu"])
        cfg=replace(cfg,ann_effort=ann_policy.initial_effort)
    if method in ("G2CUDA","G4CUDA","G5CUDA"):
        from pcap.controlled_fit import fit_controlled
        return fit_controlled(points,initial,cfg,device="cuda",use_ann=True,
            mode={"G2CUDA":"fixed","G4CUDA":"ann_only","G5CUDA":"combined"}[method],
            batch_policy=batch_policy,ann_policy=ann_policy)
    if method == "C1":
        from pcap.minibatch import fit_minibatch_cpu
        return fit_minibatch_cpu(points, initial, replace(cfg, ann_effort=None))
    if method == "C0":
        from pcap.kmeans_cpu import fit_cpu
        return fit_cpu(points, initial, replace(cfg, batch_size=None, ann_effort=None))
    if method == "G0":
        from pcap.kmeans_cuda import fit_cuda
        return fit_cuda(points, initial, replace(cfg, batch_size=None, ann_effort=None))
    if method == "G1":
        from pcap.minibatch_cuda import fit_minibatch_cuda
        return fit_minibatch_cuda(points, initial, replace(cfg, ann_effort=None))
    if method == "G3CUDA":
        from pcap.adaptive_cuda import fit_adaptive_minibatch_cuda
        return fit_adaptive_minibatch_cuda(points, initial, replace(cfg, ann_effort=None), policy=batch_policy)
    if method == "G2":
        from pcap.minibatch_ann import fit_minibatch_ann_cpu
        return fit_minibatch_ann_cpu(points, initial, replace(cfg, ann_effort=cfg.ann_effort or 64))
    if method in ("G3", "G4", "G5"):
        from pcap.adaptive_combined import fit_adaptive_minibatch
        mode = {"G3": "batch_only", "G4": "ann_only", "G5": "combined"}[method]
        cfg = replace(cfg, ann_effort=None if method == "G3" else cfg.ann_effort or 64)
        return fit_adaptive_minibatch(points, initial, cfg, mode=mode,batch_policy=batch_policy,ann_policy=ann_policy)
    raise ValueError(f"Unknown method: {method}")


def validate_result(points, result):
    from pcap.assignments_exact import assign_exact
    exact = assign_exact(points, result.centroids)
    if not np.array_equal(exact.labels, result.labels):
        raise ValueError("Final labels disagree with exact reference")
    if not np.isclose(result.sse, np.sum(exact.squared_distances), rtol=1e-9, atol=1e-9):
        raise ValueError("Final SSE disagrees with exact reference")


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    packages = {}
    for name in ("numpy", "numba", "numba-cuda", "hnswlib", "rasterio", "matplotlib", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu_conditions = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.free,memory.total,pstate,temperature.gpu,clocks.sm,utilization.gpu", "--format=csv"], text=True, stderr=subprocess.STDOUT, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        gpu_conditions = f"unavailable: {type(exc).__name__}"
    return {"gpu_conditions_before": gpu_conditions, "machine": platform.machine(),
            "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for directory in ("src", "experiments", "scripts", "native")
                              for p in sorted((ROOT / directory).rglob("*")) if p.is_file() and p.suffix in (".py",".cpp",".sh")}}



def snapshot_sources(destination, hashes):
    """Capture exact working sources before a measured run, including dirty files."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for relative,expected in hashes.items():
        source=ROOT/relative
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f"Source changed before snapshot: {relative}")
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    lock=ROOT/'requirements-lock.txt'
    if lock.exists(): shutil.copyfile(lock,destination/'requirements-lock.txt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=[m for m in METHODS if m not in ("G3CUDA", "C1", "G2CUDA", "G4CUDA", "G5CUDA", "O0")])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-models", action="store_true", help="Save measured labels, centroids and per-run JSON outside timing")
    parser.add_argument("--budget-seconds", type=float, default=120, help="Checked between fits; an active fit is not interrupted")
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--ann-effort", type=int, default=None, help="HNSW ef for CPU; candidate budget for CUDA projection ANN")
    parser.add_argument("--probe-interval", type=int, default=5)
    parser.add_argument("--policy-json", type=Path, help="Frozen batch/ann_cpu/ann_cuda policy settings")
    parser.add_argument("--scene-dir", type=Path, help="Prepared scene containing X.npy and metadata.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "benchmark_suite")
    args = parser.parse_args(argv)
    if min(args.sizes + args.k + [args.d, args.batch_size, args.max_steps, args.repeats, args.warmups]) < 1 or min(args.seeds) < 0:
        parser.error("sizes, k, d, batch, steps, repeats and warmups must be positive; seeds nonnegative")
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("budget-seconds must be finite and positive")
    policy_settings=json.loads(args.policy_json.read_text()) if args.policy_json else None
    if policy_settings is not None and (not isinstance(policy_settings,dict) or set(policy_settings)-{"batch","ann_cpu","ann_cuda"}):
        parser.error("Unknown policy configuration sections")
    # Validate before opening output files.
    FitConfig(1, 0, args.max_steps, args.tolerance,probe_interval=args.probe_interval,ann_effort=args.ann_effort)
    scene = None
    scene_metadata = None
    if args.scene_dir:
        scene = np.load(args.scene_dir / "X.npy", allow_pickle=False)
        validate_features("scene X", scene)
        scene_metadata = json.loads((args.scene_dir / "metadata.json").read_text())
        if scene_metadata.get("X_sha256") != array_hash(scene):
            raise ValueError("Prepared scene hash mismatch")
    sizes = [len(scene)] if scene is not None else args.sizes
    if any(k > n for n in sizes for k in args.k):
        parser.error("k must not exceed dataset size")
    if policy_settings:
        from pcap.adaptive_controller import BatchPolicy
        from pcap.adaptive_ann_controller import ANNEffortPolicy
        for section,options in policy_settings.items():
            if not isinstance(options,dict):
                parser.error("Policy sections must be objects")
            if section == "batch":
                for n in sizes: BatchPolicy.for_run(n,min(args.batch_size,n),**options)
            else:
                check=ANNEffortPolicy(**options)
                if args.ann_effort is not None and check.initial_effort != args.ann_effort:
                    parser.error("Policy initial effort and --ann-effort disagree")
        json.dumps(policy_settings,allow_nan=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest_path = args.output_dir / f"manifest_{stamp}.json"
    raw_path = args.output_dir / f"raw_{stamp}.jsonl"
    csv_path = args.output_dir / f"benchmark_results_{stamp}.csv"
    summary_path = args.output_dir / f"summary_{stamp}.csv"
    completed_rows = []
    suite_start = perf_counter()
    model_dir = args.output_dir / f"models_{stamp}"
    manifest = {**provenance(), "policy_settings":policy_settings, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset": "prepared_scene" if scene is not None else "synthetic_gaussian",
                "scene_metadata": scene_metadata, "method_implementations": METHODS,
                "timing_policy": "external complete fit, synchronized; warmups excluded; quality outside timer; cold warmup retained",
                "limitations": "G2-G5 CPU only; pilot, no time-to-target or held-out protocol enforced",
                "total_completed": 0, "expected_measured_runs": len(sizes) * len(args.k) * len(args.seeds) * len(args.methods) * args.repeats,
                "failures": [], "complete": False, "csv_path": str(csv_path),
                "raw_path": str(raw_path), "summary_path": str(summary_path)}
    snapshot_sources(args.output_dir / f"source_{stamp}",manifest["source_hashes"])
    fields = ["dataset_id", "case_id", "session_id", "X_sha256", "C0_sha256", "method", "implementation", "seed", "n", "d", "k", "repeat", "batch_size", "threads_requested", "threads_actual", "native_fit_s", "steps",
              "termination_reason", "convergence_satisfied", "points_processed", "final_exact_sse", "sse_per_point", "rmse", "davies_bouldin_index",
              "silhouette_score", "wall_time_s", "compute_time_s", "throughput_pixels_per_s"]
    try:
        with csv_path.open("x", newline="") as csv_file, raw_path.open("x") as raw:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for n in sizes:
                for k in args.k:
                    for seed in args.seeds:
                        points, initial = make_synthetic_dataset(n, args.d, k, seed) if scene is None else (scene, initialize_centroids(scene, k, seed=seed))
                        x_hash, c_hash = array_hash(points), array_hash(initial)
                        cfg = FitConfig(k, seed, args.max_steps, args.tolerance, min(args.batch_size, n), args.probe_interval,args.ann_effort)
                        dataset_id=hashlib.sha256(json.dumps({"X":x_hash,"scene":scene_metadata},sort_keys=True).encode()).hexdigest()
                        case_id=hashlib.sha256(json.dumps({"dataset":dataset_id,"C0":c_hash,"config":asdict(cfg),"policies":policy_settings,"omp_threads":args.omp_threads},sort_keys=True).encode()).hexdigest()
                        for phase, repeats in (("warmup", args.warmups), ("measured", args.repeats)):
                            for repeat in range(repeats):
                                # Rotate method order to reduce systematic order bias.
                                offset = (seed + repeat) % len(args.methods)
                                order = args.methods[offset:] + args.methods[:offset]
                                for method in order:
                                    try:
                                        if perf_counter()-suite_start > args.budget_seconds:
                                            raise TimeoutError("Suite budget exhausted between fits")
                                        if method in ("G0", "G1", "G3CUDA", "G2CUDA", "G4CUDA", "G5CUDA"):
                                            from numba import cuda
                                            cuda.synchronize()
                                        start = perf_counter()
                                        result = run_single_method(method,points,initial,cfg,omp_threads=args.omp_threads) if method=="O0" else run_single_method(method, points, initial, cfg,policy_settings=policy_settings) if policy_settings else run_single_method(method, points, initial, cfg)
                                        if method in ("G0", "G1", "G3CUDA", "G2CUDA", "G4CUDA", "G5CUDA"):
                                            cuda.synchronize()
                                        elapsed = perf_counter() - start
                                        validate_result(points, result)
                                        if array_hash(points) != x_hash or array_hash(initial) != c_hash:
                                            raise ValueError("Fit modified shared inputs")
                                        record = {"dataset_id":dataset_id,"case_id":case_id,"session_id":stamp,"method": method, "implementation": METHODS[method], "phase": phase,
                                                  "repeat": repeat, "seed": seed, "n": n, "d": points.shape[1], "k": k,
                                                  "X_sha256": x_hash, "C0_sha256": c_hash, "external_fit_s": elapsed,
                                                  "metadata": result.metadata, "timings": result.timings, "history": result.history}
                                        raw.write(json.dumps(record, allow_nan=False) + "\n"); raw.flush()
                                        if phase == "measured":
                                            q = evaluate_clustering_quality(points, result, seed=seed)
                                            if args.save_models:
                                                run_dir = model_dir / f"n{n}_k{k}_seed{seed}_{method}_rep{repeat}"
                                                run_dir.mkdir(parents=True, exist_ok=False)
                                                np.save(run_dir / "labels.npy", result.labels)
                                                np.save(run_dir / "centroids.npy", result.centroids)
                                                (run_dir / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False))
                                            row = {key: q[key] for key in fields if key in q}
                                            row.update(dataset_id=dataset_id,case_id=case_id,session_id=stamp,X_sha256=x_hash,C0_sha256=c_hash,method=method, implementation=METHODS[method], seed=seed, n=n,
                                                       d=points.shape[1], k=k, repeat=repeat, batch_size=result.metadata["config"]["batch_size"],
                                                       threads_requested=result.metadata.get("threads_requested"),threads_actual=result.metadata.get("threads_actual"),native_fit_s=result.timings.get("native_fit_s"),
                                                       convergence_satisfied=result.metadata.get("convergence_satisfied", result.metadata.get("termination_reason") == "tolerance"),
                                                       points_processed=result.metadata.get("points_processed", n*result.metadata["steps"]),
                                                       wall_time_s=elapsed, throughput_pixels_per_s=n / max(elapsed, 1e-12))
                                            writer.writerow(row); csv_file.flush()
                                            completed_rows.append(row)
                                            manifest["total_completed"] += 1
                                    except Exception as exc:
                                        manifest["failures"].append({"method": method, "phase": phase, "seed": seed, "n": n, "k": k,
                                                                     "error": f"{type(exc).__name__}: {exc}"})
                                        raise
        manifest["complete"] = True
    finally:
        from collections import defaultdict
        groups = defaultdict(list)
        keys = ("dataset_id","case_id","session_id","method", "implementation", "n", "d", "k", "seed")
        for row in completed_rows:
            groups[tuple(row[key] for key in keys)].append(row)
        with summary_path.open("x", newline="") as summary_file:
            summary_fields = [*keys, "completed_repeats", "expected_repeats", "mean_wall_s", "std_wall_s", "min_wall_s", "max_wall_s", "mean_sse", "std_sse"]
            writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
            writer.writeheader()
            for key, rows in groups.items():
                times = np.array([row["wall_time_s"] for row in rows])
                sses = np.array([row["final_exact_sse"] for row in rows])
                writer.writerow(dict(zip(keys, key), completed_repeats=len(rows), expected_repeats=args.repeats,
                                     mean_wall_s=float(times.mean()), std_wall_s=float(times.std()),
                                     min_wall_s=float(times.min()), max_wall_s=float(times.max()),
                                     mean_sse=float(sses.mean()), std_sse=float(sses.std())))
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        artifacts = [csv_path, raw_path, summary_path, manifest_path]
        if model_dir.exists():
            artifacts += sorted(p for p in model_dir.rglob("*") if p.is_file())
        checksum_path = args.output_dir / f"checksums_{stamp}.sha256"
        checksum_path.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(args.output_dir)}\n" for p in artifacts))
    print(f"Completed pilot: {csv_path}; raw logs: {raw_path}")
    return csv_path


if __name__ == "__main__":
    main()
