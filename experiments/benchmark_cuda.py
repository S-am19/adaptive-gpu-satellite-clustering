"""Small reproducible CPU/CUDA fit-call benchmark; no algorithm modifications.

Default pilot: N=1024/8192, D=4/14, K=4/16, three seeds, five repeats.
Results default to a unique directory under /tmp/pcap-benchmarks, outside Git.
"""

import argparse
import itertools
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import get_clock_info, perf_counter
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TIMING_KEYS = ("compute_s", "wall_s", "final_assignment_s", "kernel_s", "setup_s",
               "upload_s", "scalar_transfer_s", "download_s", "transfer_s")
RAW_FIELDS = ("timestamp_utc", "case_id", "phase", "repeat", "method", "order", "n", "d", "k", "seed",
              "max_steps", "tolerance", "data_sha256", "initial_sha256", "external_fit_s",
              "sse", "sse_per_point", "steps", "termination_reason",
              "final_centroid_displacement", "convergence_satisfied", "quality_valid",
              "quality_json", "warnings_json", "metadata_json", *TIMING_KEYS)


def array_hash(array):
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def make_synthetic(n, d, k, seed):
    """Seeded Gaussian mixture; no true-class accuracy is inferred from it."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, (k, d))
    membership = rng.integers(0, k, n)
    return np.ascontiguousarray(centers[membership] + rng.normal(0.0, 0.35, (n, d)),
                                dtype=np.float32)


def spread(values):
    values = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median_s": float(median), "q25_s": float(q25), "q75_s": float(q75),
            "iqr_s": float(q75 - q25), "min_s": float(values.min()),
            "max_s": float(values.max()), "std_s": float(values.std(ddof=1))}


def verify_quality(points, cpu, gpu, exact_assign):
    """Outside timed regions: validate final objectives, labels and matched work."""
    checks = {}
    for name, result in (("cpu", cpu), ("gpu", gpu)):
        exact = exact_assign(points, result.centroids)
        checks[f"{name}_labels_exact"] = bool(np.array_equal(result.labels, exact.labels))
        checks[f"{name}_sse_exact"] = bool(np.isclose(
            result.sse, np.sum(exact.squared_distances, dtype=np.float64), rtol=1e-10, atol=1e-10))
    checks["sse_agrees"] = bool(np.isclose(cpu.sse, gpu.sse, rtol=1e-9, atol=1e-10))
    checks["centroids_agree"] = bool(np.allclose(cpu.centroids, gpu.centroids, rtol=2e-6, atol=2e-6))
    checks["labels_agree"] = bool(np.array_equal(cpu.labels, gpu.labels))
    checks["steps_agree"] = cpu.metadata["steps"] == gpu.metadata["steps"]
    checks["termination_agrees"] = cpu.metadata["termination_reason"] == gpu.metadata["termination_reason"]
    checks["finite_outputs"] = bool(all(np.isfinite(result.centroids).all()
                                        and np.isfinite(result.sse) for result in (cpu, gpu)))
    valid = all(checks.values())
    checks["label_agreement"] = float(np.mean(cpu.labels == gpu.labels))
    checks["relative_sse_gap"] = abs(cpu.sse - gpu.sse) / max(abs(cpu.sse), np.finfo(float).tiny)
    return valid, checks


def _convergence_fields(result, tolerance):
    """Check the actual final movement, including when steps == max_steps."""
    displacement = float(result.history[-1]["centroid_displacement"])
    if not np.isfinite(displacement) or displacement < 0:
        raise ValueError("Final centroid displacement must be finite and nonnegative")
    satisfied = displacement <= tolerance
    reason = result.metadata["termination_reason"]
    if reason not in ("tolerance", "max_steps") or (reason == "tolerance") != satisfied:
        raise ValueError("Termination reason disagrees with the final convergence criterion")
    return {"final_centroid_displacement": displacement, "convergence_satisfied": satisfied}


def _timing_policy_metadata():
    """Describe explicit collection separately from automatic collection."""
    return {
        "excluded": (
            "generation, initialization, hashing, validation of outputs, "
            "explicit gc.collect(), CSV writes, warm-up calls"
        ),
        "garbage_collection": {
            "explicit_collect_outside_timer": True,
            "automatic_gc_enabled_at_start": gc.isenabled(),
            "automatic_gc_policy": "not disabled by runner; may run inside measured fit calls when enabled",
        },
    }


def summarize(rows):
    measured = [row for row in rows if row["phase"] == "measured"]
    methods = {name: [row for row in measured if row["method"] == name] for name in ("cpu", "gpu")}
    summary = {key: measured[0][key] for key in ("case_id", "n", "d", "k", "seed")}
    summary["repeats"] = len(methods["cpu"])
    valid = (len(methods["cpu"]) >= 5 and len(methods["cpu"]) == len(methods["gpu"])
             and all(row["quality_valid"] for row in measured))
    summary["quality_valid"] = valid
    for method, samples in methods.items():
        summary.update({f"{method}_{key}": value
                        for key, value in spread([row["external_fit_s"] for row in samples]).items()})
        summary[f"{method}_iteration_counts"] = json.dumps([row["steps"] for row in samples])
        summary[f"{method}_termination_reasons"] = json.dumps(sorted({row["termination_reason"]
                                                                     for row in samples}))
        summary[f"{method}_convergence_satisfied"] = all(row["convergence_satisfied"] for row in samples)
        summary[f"{method}_converged_repeats"] = sum(row["convergence_satisfied"] for row in samples)
        summary[f"{method}_final_displacement_max"] = max(row["final_centroid_displacement"]
                                                         for row in samples)
    summary["end_to_end_speedup"] = (summary["cpu_median_s"] / summary["gpu_median_s"] if valid else "")
    for key in TIMING_KEYS:
        summary[f"gpu_{key}_median"] = float(np.median([row[key] for row in methods["gpu"]]))
    summary["steps_min"] = min(row["steps"] for row in measured)
    summary["steps_max"] = max(row["steps"] for row in measured)
    summary["max_relative_sse_gap"] = max(json.loads(row["quality_json"])["relative_sse_gap"]
                                           for row in measured)
    return summary


def _command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def _available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available RAM")


def _configure_process():
    # The ELF loader reads LD_LIBRARY_PATH at process start, so re-exec before
    # importing CUDA when the WSL driver prefix needs to be installed.
    release = platform.release().lower()
    if "microsoft" in release and "wsl2" in release:
        bridge = "/usr/lib/wsl/lib"
        if not Path(bridge, "libcuda.so").is_file():
            raise RuntimeError("WSL CUDA driver bridge is missing")
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing.split(":")[0] != bridge:
            env = dict(os.environ, LD_LIBRARY_PATH=bridge + (":" + existing if existing else ""))
            os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()),
                                      *sys.argv[1:]], env)
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def case_schedule(args):
    """Yield (case, phase, repeat, CPU/GPU order); interleave outside timers."""
    cases = list(itertools.product(sorted(set(args.sizes)), sorted(set(args.dimensions)),
                                   sorted(set(args.clusters)), dict.fromkeys(args.seeds)))
    phases = (("warmup", args.warmups), ("measured", args.repeats))
    if args.interleave_sizes:
        for phase, count in phases:
            for repeat in range(count):
                ordered = cases if repeat % 2 == 0 else list(reversed(cases))
                for case in ordered:
                    index = cases.index(case)
                    order = ("cpu", "gpu") if (repeat + case[3] + index) % 2 == 0 else ("gpu", "cpu")
                    yield case, phase, repeat, order
    else:
        for case in cases:
            for phase, count in phases:
                for repeat in range(count):
                    order = ("cpu", "gpu") if (repeat + case[3]) % 2 == 0 else ("gpu", "cpu")
                    yield case, phase, repeat, order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=_positive, default=[1024, 8192])
    parser.add_argument("--dimensions", nargs="+", type=_positive, default=[4, 14])
    parser.add_argument("--clusters", nargs="+", type=_positive, default=[4, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument("--warmups", type=_positive, default=1)
    parser.add_argument("--max-steps", type=_positive, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--budget-seconds", type=_positive, default=120)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interleave-sizes", action="store_true",
                        help="warm all cases first, then alternate case order each measured round")
    args = parser.parse_args(argv)
    if args.repeats < 5:
        parser.error("at least five measured repeats are required")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    if max(args.clusters) > min(args.sizes):
        parser.error("every case must satisfy K <= N")
    return args


def main(argv=None):
    args = parse_args(argv)
    _configure_process()
    sys.path.insert(0, str(ROOT / "src"))
    from numba import cuda
    from numba.cuda.core.errors import NumbaPerformanceWarning
    from pcap.assignments_exact import assign_exact
    from pcap.contracts import FitConfig
    from pcap.initialization import initialize_centroids
    from pcap.kmeans_cpu import fit_cpu
    from pcap.kmeans_cuda import fit_cuda

    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Benchmark requires a real NVIDIA CUDA device")
    cuda.synchronize()
    started = perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_dir or Path(tempfile.gettempdir()) / "pcap-benchmarks" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    model_lines = [line for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name')]
    manifest = {
        "timestamp_utc": timestamp, "settings": vars(args) | {"output_dir": str(output)},
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "cpu_model": model_lines[0].split(':', 1)[1].strip() if model_lines else platform.processor(),
        "logical_cpus": os.cpu_count(), "available_ram_bytes": _available_ram(),
        "initial_cuda_memory_info": list(cuda.current_context().get_memory_info()),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,memory.used,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu",
                                 "--format=csv,noheader,nounits"]),
        "nvcc": _command(["nvcc", "--version"]),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("src/pcap/kmeans_cpu.py", "src/pcap/kmeans_cuda.py",
                                       "src/pcap/contracts.py", "experiments/benchmark_cuda.py")},
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NUMBA_ENABLE_CUDASIM",
            "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "timing_policy": "perf_counter around complete fit call; pre-call CUDA sync outside both timers; fit_cuda synchronizes internally",
        "clock": vars(get_clock_info("perf_counter")),
        **_timing_policy_metadata(),
        "benchmark_schema_version": 2,
        "quality_policy": "exact final labels/SSE, centroid tolerance 2e-6, CPU/GPU SSE rtol 1e-9, same steps/termination",
        "warning_policy": "ignore only small-grid NumbaPerformanceWarning occupancy hints; record other warnings",
        "memory_policy": "conservative estimated host/device buffers, not measured peak usage",
        "scope": "small synthetic pilot, not final project-scale evaluation", "status": "running",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    manifest["schedule"] = [{"case": case, "phase": phase, "repeat": repeat, "order": order}
                            for case, phase, repeat, order in case_schedule(args)]
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    summaries = []
    states = {}
    host_budget = 0.25 * manifest["available_ram_bytes"]
    print(f"Artifacts: {output}", flush=True)
    print("External full-fit medians and IQR; speedup = CPU median / GPU median", flush=True)

    def check_budget():
        if perf_counter() - started > args.budget_seconds:
            raise TimeoutError("Pilot time budget reached; completed rows are preserved")

    try:
        with (output / "raw.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for case, phase, repeat, order in case_schedule(args):
                check_budget()
                n, d, k, seed = case
                if case not in states:
                    host_estimate = 32 * n * d + 64 * n + 32 * k * d
                    device_estimate = 4 * n * d + 12 * n + 12 * k * d + 16 * k + 65536
                    retained_estimate = sum(32 * c[0] * c[1] + 64 * c[0] + 32 * c[2] * c[1]
                                            for c in states)
                    if (retained_estimate + host_estimate > host_budget
                            or host_estimate > 0.25 * _available_ram()):
                        raise MemoryError("Case exceeds conservative pilot RAM budget")
                    if device_estimate > 0.5 * cuda.current_context().get_memory_info()[0]:
                        raise MemoryError("Case exceeds conservative pilot VRAM budget")
                    points = make_synthetic(n, d, k, seed)
                    initial = initialize_centroids(points, k, seed=seed)
                    config = FitConfig(k, seed, args.max_steps, args.tolerance)
                    data_hash, initial_hash = array_hash(points), array_hash(initial)
                    case_id = f"n{n}_d{d}_k{k}_seed{seed}"
                    states[case] = (points, initial, config, data_hash, initial_hash, case_id, [])
                points, initial, config, data_hash, initial_hash, case_id, case_rows = states[case]
                pair, rows = {}, []
                try:
                    for position, method in enumerate(order):
                        gc.collect()
                        cuda.synchronize()  # Drain previous work outside either method's timer.
                        fit = fit_cpu if method == "cpu" else fit_cuda
                        measured_at = datetime.now(timezone.utc).isoformat()
                        with warnings.catch_warnings(record=True) as observed:
                            warnings.filterwarnings("ignore", message="Grid size .*GPU under-utilization.*",
                                                    category=NumbaPerformanceWarning)
                            start = perf_counter()
                            result = fit(points, initial, config)
                            elapsed = perf_counter() - start
                        pair[method] = result
                        rows.append({
                            "timestamp_utc": measured_at,
                            "case_id": case_id, "phase": phase, "repeat": repeat,
                            "method": method, "order": position, "n": n, "d": d, "k": k,
                            "seed": seed, "max_steps": args.max_steps, "tolerance": args.tolerance,
                            "data_sha256": data_hash, "initial_sha256": initial_hash,
                            "external_fit_s": elapsed, "sse": result.sse, "sse_per_point": result.sse / n,
                            "steps": result.metadata["steps"],
                            "termination_reason": result.metadata["termination_reason"],
                            **_convergence_fields(result, config.tolerance),
                            "warnings_json": json.dumps([str(item.message) for item in observed]),
                            "metadata_json": json.dumps(result.metadata, sort_keys=True),
                            **{key: result.timings.get(key, "") for key in TIMING_KEYS},
                        })
                    valid, checks = verify_quality(points, pair["cpu"], pair["gpu"], assign_exact)
                    checks["inputs_unchanged"] = (array_hash(points) == data_hash
                                                   and array_hash(initial) == initial_hash)
                    valid = valid and checks["inputs_unchanged"]
                except Exception:
                    # Keep successful calls from an interrupted pair, without claiming validation.
                    for row in rows:
                        row.update(quality_valid=False, quality_json=json.dumps({"pair_incomplete": True}))
                        writer.writerow(row)
                    handle.flush()
                    raise
                for row in rows:
                    row.update(quality_valid=valid, quality_json=json.dumps(checks, sort_keys=True))
                    writer.writerow(row)
                handle.flush()
                case_rows.extend(rows)
                if not valid:
                    raise RuntimeError(f"Quality/matched-work check failed: {case_id}: {checks}")
                if phase == "measured" and repeat == args.repeats - 1:
                    summary = summarize(case_rows)
                    summaries.append(summary)
                    print(f"{summary['case_id']}: CPU {summary['cpu_median_s']*1000:.3f} ms "
                          f"(IQR {summary['cpu_iqr_s']*1000:.3f}); "
                          f"GPU {summary['gpu_median_s']*1000:.3f} ms "
                          f"(IQR {summary['gpu_iqr_s']*1000:.3f}); "
                          f"speedup {summary['end_to_end_speedup']:.3f}x; "
                          f"steps {summary['steps_min']}..{summary['steps_max']}; "
                          f"criterion satisfied CPU={summary['cpu_convergence_satisfied']} "
                          f"GPU={summary['gpu_convergence_satisfied']}", flush=True)
                    del states[case]
                # Drop loop-local array/result references before loading the next case.
                del points, initial, config, pair, result, case_rows
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if summaries:
            with (output / "summary.csv").open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        manifest["completed_cases"] = len(summaries)
        manifest["elapsed_s"] = perf_counter() - started
        manifest["final_available_ram_bytes"] = _available_ram()
        try:
            manifest["final_cuda_memory_info"] = list(cuda.current_context().get_memory_info())
        except Exception as exc:
            # Diagnostics must not mask a fit failure or prevent its manifest being saved.
            manifest["final_cuda_memory_info"] = None
            manifest["final_cuda_memory_error"] = f"{type(exc).__name__}: {exc}"
        manifest["final_nvidia_smi"] = _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,memory.used,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu", "--format=csv,noheader,nounits"])
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return output


if __name__ == "__main__":
    main()
