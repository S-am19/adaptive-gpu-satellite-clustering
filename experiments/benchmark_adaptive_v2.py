"""Validation-only replay and controlled comparison for experimental V2.

The evaluator may derive the frozen 5% target from a converged CPU reference,
but never passes that value into any fit or controller. Existing G1/G3CUDA are
run through the project's instrumented, target-blind ``fit_controlled`` path so
all arms receive the same full-data quality-check cadence.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pcap.adaptive_controller import BatchPolicy
from pcap.adaptive_v2_controller import AdaptiveV2Policy
from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig, validate_features
from pcap.initialization import initialize_centroids
from pcap.kmeans_cpu import fit_cpu
from pcap.minibatch_decayed_cuda import (
    fit_adaptive_decayed_minibatch_cuda,
    fit_decayed_minibatch_cuda,
)
from pcap.controlled_fit import fit_controlled


DEFAULT_REPLAY = ROOT / "outputs/20261026-validation-decayed-count-k16-b8192-seed0"
DEFAULT_SCENE = ROOT / "outputs/20261025-independent-holdouts/validation512"
METHODS = ("G0", "A_G1", "B_G3CUDA", "C_DECAYED_FIXED", "D_ADAPTIVE_V2")


def digest_bytes(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_snapshot():
    query = "name,driver_version,memory.free,memory.total,pstate,temperature.gpu,clocks.current.sm,utilization.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv"],
            check=True, text=True, capture_output=True, timeout=10)
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def ram_snapshot():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, val = line.split(":", 1)
                values[key] = val.strip()
    except OSError as exc:
        values["error"] = str(exc)
    return values


def environment():
    packages = {}
    for name in ("numpy", "numba", "numba-cuda", "llvmlite", "cuda-bindings", "cuda-core", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        nvcc = subprocess.run(["nvcc", "--version"], check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip().splitlines()[-1]
    except Exception as exc:
        nvcc = f"unavailable: {type(exc).__name__}: {exc}"
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                nvcc=nvcc, cuda_simulator=os.environ.get("NUMBA_ENABLE_CUDASIM", "0"))


def source_hashes():
    paths = ["experiments/benchmark_adaptive_v2.py", "src/pcap/minibatch_decayed_cuda.py",
             "src/pcap/adaptive_v2_controller.py", "src/pcap/minibatch_cuda.py",
             "src/pcap/adaptive_cuda.py", "src/pcap/controlled_fit.py",
             "src/pcap/contracts.py", "src/pcap/initialization.py"]
    return {rel: digest_file(ROOT / rel) for rel in paths}


def write_inventory(folder):
    entries = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{digest_file(path)}  {path.relative_to(folder).as_posix()}\n")
    (folder / "checksums.sha256").write_text("".join(entries))


def write_measurements(path, records):
    """Write rows whose timing dictionaries may have method-specific fields."""
    fieldnames = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def verify_scene(scene_dir):
    scene_dir = Path(scene_dir).resolve()
    points_path = scene_dir / "X.npy"
    meta_path = scene_dir / "metadata.json"
    X = np.load(points_path, mmap_mode="r", allow_pickle=False)
    validate_features("X", X)
    meta = json.loads(meta_path.read_text())
    x_hash = digest_bytes(X)
    if meta.get("X_sha256") != x_hash:
        raise ValueError("Prepared validation scene X hash does not match metadata")
    split = str(meta.get("split", "")).lower()
    if split and split not in {"validation", "val"}:
        raise ValueError(f"Expected the validation split, got {split!r}")
    return scene_dir, X, meta, x_hash


def load_frozen_target(scene_dir, X, seed, k, reference_steps, quality_gap):
    C0 = initialize_centroids(X, k, seed=seed)
    cfg = FitConfig(k, seed, reference_steps, 1e-6)
    ref = fit_cpu(np.ascontiguousarray(X), C0, cfg)
    if ref.metadata.get("termination_reason") != "tolerance":
        raise RuntimeError(f"Seed {seed}: exact CPU reference did not converge")
    target = ref.sse * (1.0 + quality_gap)
    return C0, ref, target


def target_metrics(quality, target):
    crossing = next((row for row in quality if row["full_data_sse"] <= target), None)
    return (None if crossing is None else float(crossing["elapsed_s"]), crossing is not None)


def result_arrays(folder, method, seed, result):
    path = folder / f"arrays_{method}_seed{seed}.npz"
    np.savez_compressed(path, centroids=result.centroids, labels=result.labels)
    return path.name


def baseline_fit(method, X, C0, cfg, quality_interval, batch_policy, v2_policy,
                 observation_budget=None, quality_observation_checkpoints=()):
    if method == "G0":
        exact_cfg = FitConfig(cfg.k, cfg.seed, max(cfg.max_steps, 1000), cfg.tolerance)
        return fit_controlled(X, C0, exact_cfg, device="cuda", mode="full",
                              evaluation_interval=(quality_interval or 100), target_sse=None)
    if method == "A_G1":
        return fit_controlled(X, C0, cfg, device="cuda", mode="fixed",
                              evaluation_interval=quality_interval, target_sse=None)
    if method == "B_G3CUDA":
        return fit_controlled(X, C0, cfg, device="cuda", mode="batch_only",
                              batch_policy=batch_policy,
                              evaluation_interval=quality_interval, target_sse=None)
    if method == "C_DECAYED_FIXED":
        return fit_decayed_minibatch_cuda(
            X, C0, cfg, quality_interval=quality_interval,
            observation_budget=observation_budget,
            quality_observation_checkpoints=quality_observation_checkpoints)
    if method == "D_ADAPTIVE_V2":
        return fit_adaptive_decayed_minibatch_cuda(
            X, C0, cfg, policy=v2_policy, quality_interval=quality_interval,
            observation_budget=observation_budget,
            quality_observation_checkpoints=quality_observation_checkpoints)
    raise ValueError(f"Unknown method: {method}")


def run_replay(args, output, manifest):
    scene_dir, X, scene_meta, x_hash = verify_scene(args.scene_dir)
    archive = Path(args.replay_archive).resolve()
    reference = json.loads((archive / "summary.json").read_text())
    if x_hash != reference["x_sha256"]:
        raise ValueError("Validation scene differs from archived CPU decayed diagnostic")
    k, seed = int(reference["k"]), int(reference["seed"])
    C0 = initialize_centroids(X, k, seed=seed)
    c_hash = digest_bytes(C0)
    if c_hash != reference["c0_sha256"]:
        raise ValueError("Reconstructed C0 differs from archived diagnostic")
    indices = np.load(archive / "batch_indices.npy", mmap_mode="r", allow_pickle=False)
    if digest_bytes(indices) != reference["batch_indices_sha256"]:
        raise ValueError("Archived batch-index array-content checksum does not match summary")
    steps, batch = int(reference["max_updates"]), int(reference["batch_size"])
    cfg = FitConfig(k, seed, steps, 1e-6, batch_size=batch, probe_interval=10)
    # Warm kernel compilation separately; exclude from the matched replay result.
    warm_cfg = FitConfig(k, seed, 2, cfg.tolerance, batch_size=batch, probe_interval=1)
    warm = fit_decayed_minibatch_cuda(X, C0, warm_cfg, quality_interval=2)
    started = perf_counter()
    result = fit_decayed_minibatch_cuda(X, C0, cfg, quality_interval=100,
                                        batch_indices=indices)
    elapsed = perf_counter() - started
    archived_rows = {}
    with (archive / "checkpoint_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] == "decayed_count":
                archived_rows[int(row["step"])] = float(row["full_sse"])
    gpu_rows = {int(row["step"]): float(row["full_data_sse"])
                for row in result.metadata["quality_history"]}
    common = sorted(set(archived_rows) & set(gpu_rows))
    differences = {str(step): dict(cpu_sse=archived_rows[step], cuda_sse=gpu_rows[step],
                         abs_diff=abs(gpu_rows[step]-archived_rows[step])) for step in common}
    tolerance = args.replay_rtol * max(abs(v) for v in archived_rows.values()) + args.replay_atol
    max_diff = max((abs(gpu_rows[s]-archived_rows[s]) for s in common), default=float("inf"))
    matched = len(common) == len(archived_rows) and max_diff <= tolerance
    arrays = result_arrays(output, "C_REPLAY", seed, result)
    config_record = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    payload = dict(phase="replay", method="C_DECAYED_FIXED_CUDA_REPLAY", seed=seed,
                   X_sha256=x_hash, C0_sha256=c_hash, config=config_record,
                   archived_cpu_target_evaluation_only=reference["target_sse_frozen"],
                   target_passed_to_fit=False, warmup=dict(steps=warm.metadata["steps"],
                   wall_s=warm.timings["wall_s"]), external_wall_s=elapsed,
                   quality_checkpoint_comparison=differences,
                   max_abs_sse_difference=max_diff, allowed_abs_difference=tolerance,
                   replay_matches=matched, result_arrays=arrays,
                   metadata=result.metadata, timings=result.timings)
    (output / "replay_result.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    manifest.update(scene=scene_meta, X_sha256=x_hash, C0_sha256=c_hash,
                    replay_reference=str(archive), replay_matches=matched,
                    replay_max_abs_sse_difference=max_diff,
                    replay_allowed_abs_difference=tolerance,
                    replay_steps=steps, replay_batch_size=batch)
    return matched


def run_comparison(args, output, manifest):
    scene_dir, X_view, scene_meta, x_hash = verify_scene(args.scene_dir)
    X = np.ascontiguousarray(X_view)
    n = len(X)
    if args.k > n or args.batch_size > n:
        raise ValueError("K and batch size must not exceed N")
    import numba.cuda as cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Comparison requires real CUDA; simulation and fallback are disabled")
    before_gpu, before_ram = gpu_snapshot(), ram_snapshot()
    device = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    capacity = args.maximum_batch
    estimated = (4*n*X.shape[1] + 4*args.k*X.shape[1]*2 + 16*args.k*X.shape[1]
                 + 16*args.k + 4*capacity*2 + 8*capacity
                 + 4*min(n,1024)*2 + 8*min(n,1024) + 12*n
                 + 8*((n+255)//256 + (capacity+255)//256 + 16 + 3))
    if estimated > 0.8*free:
        raise MemoryError(f"Estimated V2 resident buffers {estimated} exceed 80% of free VRAM {free}")
    methods = list(args.methods)
    policy_data = json.loads((ROOT/"configs/validation_tuning_policies.json").read_text())["batch"]
    batch_policy_settings = dict(policy_data)
    batch_policy_settings.update(minimum=args.minimum_batch, maximum=args.maximum_batch)
    records, references = [], []
    manifest.update(scene=scene_meta, X_sha256=x_hash, device=dict(name=str(device.name),
                    compute_capability=tuple(device.compute_capability),
                    free_memory_bytes=int(free), total_memory_bytes=int(total)),
                    gpu_conditions_before=before_gpu, ram_before=before_ram,
                    estimated_v2_device_bytes=int(estimated), methods=methods,
                    protocol=dict(k=args.k, seeds=args.seeds, initial_batch=args.batch_size,
                    minimum_batch=args.minimum_batch, maximum_batch=args.maximum_batch,
                    max_steps=args.max_steps, tolerance=args.tolerance,
                    probe_interval=args.probe_interval, quality_interval=args.quality_interval,
                    observation_budget=args.observation_budget,
                    quality_observation_checkpoints=args.quality_observation_checkpoints,
                    quality_gap=args.quality_gap, warmups=1, measured_repeats=1,
                    decay_half_life_passes=10, target_policy="evaluation only; never passed to fit/controller"),
                    g3cuda_policy=batch_policy_settings, v2_policy=dict(
                    minimum=args.minimum_batch, maximum=args.maximum_batch,
                    initial_batch=args.batch_size, deterioration=0.01,
                    stagnation_improvement=0.001, progress_improvement=0.005,
                    quiet_movement=0.001, quiet_label_change=0.005,
                    required_checkpoints=3), failures=[], complete=False)
    batch_policy = BatchPolicy.for_run(n, args.batch_size, **batch_policy_settings)
    v2_policy = AdaptiveV2Policy(
        minimum=args.minimum_batch, maximum=args.maximum_batch, initial_batch=args.batch_size)
    timer_start = perf_counter()

    def budget_check():
        if perf_counter()-timer_start > args.budget_seconds:
            raise TimeoutError("Adaptive V2 comparison budget exceeded between fits")

    raw_path = output / "raw.jsonl"
    with raw_path.open("x") as raw:
        # One warm-up per path; warmups are recorded but excluded from the summary.
        warm_seed = args.seeds[0]
        warm_c, _, _ = load_frozen_target(scene_dir, X, warm_seed, args.k, args.reference_steps, args.quality_gap)
        for method in methods:
            budget_check()
            steps = 2 if method == "G0" else 2
            warm_cfg = FitConfig(args.k, warm_seed, steps, args.tolerance,
                                 None if method == "G0" else args.batch_size,
                                 args.probe_interval)
            started = perf_counter()
            result = baseline_fit(method, X, warm_c, warm_cfg,
                                  min(args.quality_interval, 2), batch_policy, v2_policy)
            cuda.synchronize()
            raw.write(json.dumps(dict(phase="warmup", method=method, seed=warm_seed,
                          external_wall_s=perf_counter()-started, timings=result.timings,
                          metadata=result.metadata), allow_nan=False)+"\n")
            raw.flush()

        for seed in args.seeds:
            budget_check()
            C0, reference, target = load_frozen_target(
                scene_dir, X, seed, args.k, args.reference_steps, args.quality_gap)
            c_hash = digest_bytes(C0)
            references.append(dict(seed=seed, C0_sha256=c_hash,
                                   exact_cpu_sse=reference.sse,
                                   exact_cpu_steps=reference.metadata["steps"],
                                   frozen_target_sse=target,
                                   evaluation_only=True))
            order = methods[seed % len(methods):] + methods[:seed % len(methods)]
            for method in order:
                budget_check()
                cfg = FitConfig(args.k, seed,
                    args.g0_max_steps if method == "G0" else args.max_steps,
                    args.tolerance, None if method == "G0" else args.batch_size,
                    args.probe_interval)
                cuda.synchronize()
                started = perf_counter()
                result = baseline_fit(method, X, C0, cfg, args.quality_interval,
                                      batch_policy, v2_policy,
                                      observation_budget=(args.observation_budget
                                          if method in {"C_DECAYED_FIXED", "D_ADAPTIVE_V2"} else None),
                                      quality_observation_checkpoints=(args.quality_observation_checkpoints
                                          if method in {"C_DECAYED_FIXED", "D_ADAPTIVE_V2"} else ()))
                cuda.synchronize()
                external = perf_counter()-started
                exact = assign_exact(X, result.centroids)
                if not np.array_equal(result.labels, exact.labels):
                    raise AssertionError(f"{method} final labels are not exact")
                independent_sse = float(exact.squared_distances.sum(dtype=np.float64))
                if not np.isclose(result.sse, independent_sse, rtol=2e-10, atol=2e-8):
                    raise AssertionError(f"{method} result SSE does not match independent exact SSE")
                if digest_bytes(X) != x_hash or digest_bytes(C0) != c_hash:
                    raise AssertionError("A fit mutated shared X or C0")
                quality = result.metadata.get("quality_history", [])
                crossing_s, hit = target_metrics(quality, target)
                sizes = [int(row["batch_size"]) for row in result.history if "batch_size" in row]
                changes = sum(a != b for a,b in zip(sizes, sizes[1:]))
                record = dict(method=method, seed=seed, phase="measured",
                    execution_device=result.metadata.get("execution_device", "cuda"),
                    X_sha256=x_hash, C0_sha256=c_hash,
                    frozen_target_sse_evaluation_only=target,
                    target_passed_to_fit=False, external_wall_s=external,
                    first_target_crossing_s=crossing_s, target_satisfied=hit,
                    final_exact_sse=result.sse,
                    final_sse_relative_to_exact_reference=result.sse/reference.sse-1.0,
                    steps=result.metadata.get("steps"),
                    termination_reason=result.metadata.get("termination_reason"),
                    convergence_satisfied=result.metadata.get("convergence_satisfied"),
                    points_processed=result.metadata.get("points_processed", 0),
                    batch_size_transitions=changes,
                    batch_sizes=sizes,
                    quality_history=quality,
                    update_history=result.history,
                    timings=result.timings,
                    metadata=result.metadata,
                    arrays_file=result_arrays(output, method, seed, result))
                raw.write(json.dumps(record, allow_nan=False)+"\n")
                raw.flush()
                row = {k: record[k] for k in (
                    "method", "seed", "execution_device", "external_wall_s",
                    "first_target_crossing_s", "target_satisfied", "final_exact_sse",
                    "final_sse_relative_to_exact_reference", "steps", "termination_reason",
                    "convergence_satisfied", "points_processed", "batch_size_transitions")}
                for key, value in result.timings.items():
                    row[f"timing_{key}"] = value
                records.append(row)
    manifest["references"] = references
    manifest["gpu_conditions_after"] = gpu_snapshot()
    manifest["ram_after"] = ram_snapshot()
    manifest["elapsed_s"] = perf_counter()-timer_start
    manifest["completed_measured_fits"] = len(records)
    manifest["complete"] = len(records) == len(args.seeds)*len(methods)
    write_measurements(output/"measurements.csv", records)
    if not manifest["complete"]:
        raise RuntimeError("Comparison ended without all planned measured fits")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("replay", "compare"), required=True)
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replay-archive", type=Path, default=DEFAULT_REPLAY)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--minimum-batch", type=int, default=256)
    p.add_argument("--maximum-batch", type=int, default=65536)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--g0-max-steps", type=int, default=1000)
    p.add_argument("--reference-steps", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--probe-interval", type=int, default=10)
    p.add_argument("--quality-interval", type=int, default=100)
    p.add_argument("--observation-budget", type=int, default=None,
                   help="exact per-fit processed-observation cap for decayed Mini-Batch methods")
    p.add_argument("--quality-observation-checkpoints", type=int, nargs="*", default=[],
                   help="full-data SSE checkpoints at shared processed-observation thresholds")
    p.add_argument("--quality-gap", type=float, default=0.05)
    p.add_argument("--budget-seconds", type=float, default=900)
    p.add_argument("--replay-rtol", type=float, default=2e-7)
    p.add_argument("--replay-atol", type=float, default=2e-5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    args.scene_dir = args.scene_dir if args.scene_dir.is_absolute() else ROOT/args.scene_dir
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.mode == "compare" and (min(args.k,args.batch_size,args.maximum_batch,args.max_steps,
            args.g0_max_steps,args.reference_steps,args.probe_interval) < 1
            or args.quality_interval < 0
            or min(args.seeds)<0 or args.minimum_batch<1 or args.minimum_batch>args.batch_size
            or args.maximum_batch<args.batch_size or args.budget_seconds<=0):
        raise SystemExit("Invalid comparison configuration")
    if args.mode == "compare":
        if not args.methods:
            raise SystemExit("At least one method must be selected")
        if args.observation_budget is not None:
            if args.observation_budget < 1:
                raise SystemExit("observation budget must be positive")
            if set(args.methods) - {"G0", "C_DECAYED_FIXED", "D_ADAPTIVE_V2"}:
                raise SystemExit("Observation-budget comparisons support only G0 and decayed methods")
            checkpoints = args.quality_observation_checkpoints
            if (not checkpoints or checkpoints != sorted(set(checkpoints))
                    or checkpoints[0] < 1 or checkpoints[-1] > args.observation_budget):
                raise SystemExit("Provide strictly increasing positive quality checkpoints within the observation budget")
        elif args.quality_observation_checkpoints:
            raise SystemExit("Quality observation checkpoints require --observation-budget")
        elif args.quality_interval == 0:
            raise SystemExit("quality interval may be zero only with observation-budget checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
        git_head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                                capture_output=True,text=True).stdout.strip(),
        git_status=subprocess.run(["git","status","--short","--branch"],cwd=ROOT,
                                  check=True,capture_output=True,text=True).stdout.strip(),
        source_hashes=source_hashes(), environment=environment(),
        gpu_conditions_before=gpu_snapshot(), ram_before=ram_snapshot(),
        target_visibility="evaluation-only; target is not an argument to any fitting or controller function",
        failures=[], complete=False)
    try:
        if args.mode == "replay":
            if run_replay(args, args.output_dir, manifest):
                manifest["complete"] = True
            else:
                manifest["failures"].append("CUDA checkpoint SSE exceeded replay tolerance")
        else:
            run_comparison(args, args.output_dir, manifest)
    except Exception as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["gpu_conditions_after"] = gpu_snapshot()
        manifest["ram_after"] = ram_snapshot()
        safe_json(args.output_dir/"manifest.json", manifest)
        write_inventory(args.output_dir)
    print(f"Adaptive V2 artifacts: {args.output_dir}")
    if not manifest["complete"]:
        raise RuntimeError("Adaptive V2 experiment did not complete successfully")
    return args.output_dir


if __name__ == "__main__":
    main()
