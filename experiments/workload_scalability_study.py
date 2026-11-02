"""Controlled real-Sentinel-2 workload scaling study for CUDA Mini-Batch methods.

The primary comparison keeps cumulative and decayed update families separate.
It uses existing nested 31UFS development crops, exact CUDA reference objectives,
and a fixed observation budget for every Mini-Batch fit. Output is append-only:
choose a new output directory for each invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIZES = {
    262_144: ROOT / "outputs/20261024-plan-real-prepared/maastricht512",
    1_048_548: ROOT / "outputs/20261024-plan-real-prepared/maastricht1024",
    4_194_265: ROOT / "outputs/20261024-plan-real-prepared/maastricht2048",
}
KS = (8, 16, 32)
BATCHES = (8_192, 16_384, 32_768, 65_536)
OBSERVATION_BUDGET = 40_960_000
CHECKPOINTS = tuple(range(8_192_000, OBSERVATION_BUDGET + 1, 8_192_000))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            fields = next(csv.reader(stream))
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def _hardware() -> dict:
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") not in ("0", ""):
        raise RuntimeError("CUDA simulator is enabled; refusing simulated execution")
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required; CPU/simulator fallback is disabled")
    device = cuda.get_current_device()
    smi = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,pstate,temperature.gpu,utilization.gpu,clocks.current.sm",
        "--format=csv,noheader"], text=True, timeout=10).strip()
    free, total = cuda.current_context().get_memory_info()
    return {"device_name": str(device.name), "compute_capability": list(device.compute_capability),
            "cuda_simulator": bool(cuda.config.ENABLE_CUDASIM), "cuda_memory_free_bytes": int(free),
            "cuda_memory_total_bytes": int(total), "nvidia_smi": smi}


def _packages() -> dict:
    names = ("numpy", "numba", "numba-cuda", "llvmlite", "cuda-bindings", "cuda-core", "rasterio")
    output = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def _system_memory() -> dict:
    values = {}
    try:
        with Path("/proc/meminfo").open() as stream:
            for line in stream:
                key, raw = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError):
        return {"available": False}
    return values


def _gpu_workspace_estimate(n: int, d: int, k: int, batch: int = 65_536) -> int:
    # Conservative common peak: resident float32 X, final int32 labels and
    # float64 distances, maximum batch buffers, probe, centroids/statistics and
    # reduction scratch. This is below 200 MiB for the largest prepared crop.
    return (4*n*d + 12*n + 16*batch + 13*1024 + 64*k*d + 64*k
            + 8*((n+255)//256 + (batch+255)//256 + 32))


def _target_sse(reference_sse: float) -> float:
    """The frozen evaluation rule, detached from controller inputs."""
    if not np.isfinite(reference_sse) or reference_sse <= 0:
        raise ValueError("exact reference SSE must be finite and positive")
    return float(reference_sse * 1.05)


def _compact_schedule(history: list[dict]) -> list[dict]:
    """Keep only size transitions/controller events in the run-level CSV.

    Full per-update schedules and sampled-index hashes are stored in dedicated
    CSV ledgers so individual_runs.csv remains routinely parseable.
    """
    result, previous = [], None
    for row in history:
        size = row.get("actual_batch_size", row.get("batch_size"))
        decision = row.get("controller_decision")
        if size is None:
            continue
        if size != previous or decision is not None:
            result.append({"step": row.get("step"), "batch_size": size,
                           "points_processed": row.get("points_processed"),
                           "controller_decision": decision,
                           "next_batch_size": row.get("next_batch_size")})
        previous = size
    return result


def _git() -> dict:
    def call(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"branch": call("branch", "--show-current"), "commit": call("rev-parse", "HEAD"),
            "status_porcelain": call("status", "--porcelain")}


def _method_specs(n: int) -> list[dict]:
    specs = [{"method": f"CUM_FIXED_B{b}", "family": "cumulative", "kind": "fixed", "batch_size": b}
             for b in BATCHES]
    specs.append({"method": "QGCUDA", "family": "cumulative", "kind": "quality_guard", "batch_size": 8192})
    specs += [{"method": f"DECAY_FIXED_B{b}", "family": "decayed", "kind": "fixed_decay", "batch_size": b}
              for b in BATCHES]
    specs.append({"method": "ADAPTIVE_V2", "family": "decayed", "kind": "adaptive_v2", "batch_size": 8192})
    return specs


def _call_mini(X, C0, seed, k, spec, *, budget, max_steps):
    from pcap.contracts import FitConfig
    config = FitConfig(k, seed, max_steps, 1e-6, spec["batch_size"], 10)
    checkpoints = tuple(cp for cp in CHECKPOINTS if cp <= budget)
    if spec["family"] == "cumulative":
        from pcap.budgeted_minibatch import fit_budgeted_minibatch
        if spec["kind"] == "quality_guard":
            from pcap.quality_guard_controller import QualityGuardPolicy
            policy = QualityGuardPolicy(minimum_batch=4096, maximum_batch=65536,
                initial_batch=8192, deterioration=.01, stagnation_improvement=.001,
                progress_improvement=.005, movement_limit=.001, label_change_limit=.005,
                oscillation_change=.001, required_checkpoints=3, dwell_checkpoints=2,
                adjustment_factor=2)
            return fit_budgeted_minibatch(X, C0, config, device="cuda", observation_budget=budget,
                quality_guard_policy=policy, evaluation_interval=budget // 5,
                disjoint_probe=True, diagnostic_trace=False)
        return fit_budgeted_minibatch(X, C0, config, device="cuda", observation_budget=budget,
            evaluation_interval=budget // 5, disjoint_probe=True, diagnostic_trace=False)

    from pcap.minibatch_decayed_cuda import fit_adaptive_decayed_minibatch_cuda, fit_decayed_minibatch_cuda
    if spec["kind"] == "adaptive_v2":
        from pcap.adaptive_v2_controller import AdaptiveV2Policy
        policy = AdaptiveV2Policy.for_run(len(X), 8192, minimum=4096, maximum=65536)
        result = fit_adaptive_decayed_minibatch_cuda(X, C0, config, policy=policy,
            quality_interval=0, observation_budget=budget,
            quality_observation_checkpoints=checkpoints)
    else:
        result = fit_decayed_minibatch_cuda(X, C0, config, quality_interval=0,
            observation_budget=budget, quality_observation_checkpoints=checkpoints)
    return result, result.metadata.get("quality_history", []), []


def _run_warmup(X, seed=0, k=16):
    """Short excluded warm-up for each CUDA execution path and JIT kernel family."""
    from pcap.contracts import FitConfig
    from pcap.kmeans_cuda import fit_cuda
    from pcap.initialization import initialize_centroids
    C0 = initialize_centroids(X, k, seed=seed)
    output = []
    start = perf_counter()
    result = fit_cuda(X, C0, FitConfig(k, seed, 2, 1e-6))
    output.append({"method": "G0", "wall_s": perf_counter() - start,
                   "excluded_from_measurements": True, "steps": result.metadata["steps"]})
    for spec in _method_specs(len(X)):
        batch = spec["batch_size"]
        budget = 2 * batch
        max_steps = max(4, (budget + 4095) // 4096 + 2)
        warm_spec = dict(spec)
        if warm_spec["kind"] == "quality_guard":
            # two short updates warm assignment/update/probe code without needing
            # the full controller's three-checkpoint decision horizon.
            from pcap.budgeted_minibatch import fit_budgeted_minibatch
            from pcap.contracts import FitConfig
            from pcap.quality_guard_controller import QualityGuardPolicy
            policy = QualityGuardPolicy(4096, 65536, 8192)
            cfg = FitConfig(k, seed, max_steps, 1e-6, 8192, 1)
            start = perf_counter()
            r, _, _ = fit_budgeted_minibatch(X, C0, cfg, device="cuda", observation_budget=budget,
                quality_guard_policy=policy, evaluation_interval=budget, disjoint_probe=True)
        else:
            start = perf_counter()
            r, _, _ = _call_mini(X, C0, seed, k, warm_spec, budget=budget, max_steps=max_steps)
        output.append({"method": spec["method"], "wall_s": perf_counter() - start,
                       "excluded_from_measurements": True,
                       "steps": r.metadata.get("steps") if hasattr(r, "metadata") else None})
    return output


def _read_data() -> dict[int, tuple[np.ndarray, dict]]:
    values = {}
    for expected_n, folder in SIZES.items():
        path = folder / "X.npy"
        meta_path = folder / "metadata.json"
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Required genuine prepared scene missing: {path}")
        X = np.load(path, allow_pickle=False)
        scene = json.loads(meta_path.read_text())
        if X.ndim != 2 or X.shape[1] != 4 or X.dtype != np.float32 or not X.flags.c_contiguous:
            raise ValueError(f"Invalid Sentinel-2 feature array: {path}: {X.shape}/{X.dtype}")
        if len(X) != expected_n or not np.isfinite(X).all():
            raise ValueError(f"Unexpected N or nonfinite data in {path}: {X.shape}")
        if scene.get("scene_id") != "S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE":
            raise ValueError(f"Unexpected source scene in {meta_path}")
        if scene.get("standardization_source") != "declared_training_crops":
            raise ValueError("Expected development-fitted standardization")
        values[expected_n] = (X, scene)
    return values


def _existing_exact_reference(n: int, k: int, seed: int, x_hash: str, c_hash: str):
    """Reuse archived converged exact-CPU SSE only when both input hashes match."""
    path = ROOT / "outputs/20261026-real-quality-matrix-completion/individual_runs.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            if (row.get("method") == "C0" and row.get("n") == str(n) and row.get("k") == str(k)
                    and row.get("seed") == str(seed) and row.get("x_sha256") == x_hash
                    and row.get("c0_sha256") == c_hash):
                return {"sse": float(row["final_exact_sse"]), "source": str(path),
                        "termination_reason": row.get("termination_reason"),
                        "convergence_satisfied": row.get("convergence_satisfied"),
                        "record_sha256": _sha_bytes(json.dumps(row, sort_keys=True).encode())}
    return None


def _summarize(out: Path, runs: list[dict], checkpoints: list[dict]) -> None:
    summary = []
    groups = sorted({(r["method"], r["family"], r["n"], r["k"]) for r in runs})
    for method, family, n, k in groups:
        rows = [r for r in runs if (r["method"], r["family"], r["n"], r["k"]) == (method, family, n, k)]
        walls = np.asarray([float(r["external_wall_s"]) for r in rows])
        sses = np.asarray([float(r["final_exact_sse"]) for r in rows])
        summary.append({"method": method, "family": family, "n": n,
            "k": k, "runs": len(rows), "median_wall_s": float(np.median(walls)),
            "wall_q25_s": float(np.quantile(walls, .25)), "wall_q75_s": float(np.quantile(walls, .75)),
            "median_final_exact_sse": float(np.median(sses)),
            "median_relative_gap_to_G0_reference": float(np.median([r["relative_gap_to_reference"] for r in rows])),
            "target_hits": int(sum(r["target_hit"] for r in rows)),
            "target_hit_rate": float(np.mean([r["target_hit"] for r in rows])),
            "median_observations": float(np.median([r["observations_processed"] for r in rows])),
            "termination_reasons": ";".join(sorted({r["termination_reason"] for r in rows}))})
    _append_csv(out / "final_summary/method_summary.csv", summary)
    # Per-cell best fixed rows are selected only among observed fixed sizes; the
    # target gate is explicit and misses are not described as time-to-quality.
    cells = sorted({(r["n"], r["k"], r["family"]) for r in runs if r["family"] in ("cumulative", "decayed")})
    best_rows = []
    for n, k, family in cells:
        subset = [r for r in runs if r["n"] == n and r["k"] == k and r["family"] == family
                  and r["kind"] in ("fixed", "fixed_decay")]
        grouped = {}
        for row in subset:
            grouped.setdefault(row["method"], []).append(row)
        candidates = []
        for method, rows in grouped.items():
            hit_rate = float(np.mean([bool(r["target_hit"]) for r in rows]))
            candidates.append({"method": method, "hit_rate": hit_rate,
                "median_wall": float(np.median([float(r["external_wall_s"]) for r in rows])),
                "median_sse": float(np.median([float(r["final_exact_sse"]) for r in rows])),
                "runs": len(rows)})
        if candidates:
            top_hit_rate = max(c["hit_rate"] for c in candidates)
            tied = [c for c in candidates if c["hit_rate"] == top_hit_rate]
            if top_hit_rate > 0:
                selected = min(tied, key=lambda c: (c["median_wall"], c["median_sse"]))
                rule = "highest target-hit rate; then fastest median among fixed candidates"
            else:
                selected = min(tied, key=lambda c: (c["median_sse"], c["median_wall"]))
                rule = "no fixed candidate hit target; lowest median SSE, then fastest (not quality-matched)"
            best_rows.append({"n": n, "k": k, "family": family, "best_fixed_method": selected["method"],
                "selection_rule": rule, "best_fixed_target_hit_rate": selected["hit_rate"],
                "best_fixed_median_wall_s": selected["median_wall"],
                "best_fixed_median_sse": selected["median_sse"], "measured_runs": selected["runs"]})
    _append_csv(out / "final_summary/best_fixed_by_cell.csv", best_rows)
    _append_csv(out / "final_summary/target_attainment_by_cell.csv", [
        {"method": method, "family": family, "n": n, "k": k,
         "runs": len([r for r in runs if (r["method"], r["family"], r["n"], r["k"]) == (method, family, n, k)]),
         "target_hits": sum(bool(r["target_hit"]) for r in runs
                            if (r["method"], r["family"], r["n"], r["k"]) == (method, family, n, k)),
         "target_hit_rate": float(np.mean([bool(r["target_hit"]) for r in runs
                            if (r["method"], r["family"], r["n"], r["k"]) == (method, family, n, k)]))}
        for method, family, n, k in groups])
    adaptive_pairs = []
    for best in best_rows:
        adaptive_name = "QGCUDA" if best["family"] == "cumulative" else "ADAPTIVE_V2"
        adaptive = [r for r in runs if r["method"] == adaptive_name and r["family"] == best["family"]
                    and r["n"] == best["n"] and r["k"] == best["k"]]
        if not adaptive:
            continue
        fixed = [r for r in runs if r["method"] == best["best_fixed_method"]
                 and r["n"] == best["n"] and r["k"] == best["k"] and r["family"] == best["family"]]
        adaptive_pairs.append({"n": best["n"], "k": best["k"], "family": best["family"],
            "adaptive_method": adaptive_name, "fixed_method": best["best_fixed_method"],
            "median_adaptive_wall_s": float(np.median([r["external_wall_s"] for r in adaptive])),
            "median_fixed_wall_s": float(np.median([r["external_wall_s"] for r in fixed])),
            "adaptive_over_fixed_wall_ratio": float(np.median([r["external_wall_s"] for r in adaptive])) /
                float(np.median([r["external_wall_s"] for r in fixed])),
            "adaptive_target_hit_rate": float(np.mean([r["target_hit"] for r in adaptive])),
            "fixed_target_hit_rate": float(np.mean([r["target_hit"] for r in fixed])),
            "median_adaptive_sse": float(np.median([r["final_exact_sse"] for r in adaptive])),
            "median_fixed_sse": float(np.median([r["final_exact_sse"] for r in fixed])),
            "quality_matched_time_comparison": bool(all(r["target_hit"] for r in adaptive + fixed)),
            "interpretation": "pilot descriptive only; no inferential claim from one seed/repetition"})
    _append_csv(out / "final_summary/adaptive_vs_best_fixed.csv", adaptive_pairs)
    _write_figures(out, runs, checkpoints)


def _write_figures(out: Path, runs: list[dict], checkpoints: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "figures/plot_failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        return
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for k in KS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for method in sorted({r["method"] for r in runs if r["k"] == k}):
            rows = [r for r in runs if r["method"] == method and r["k"] == k]
            by_n = {}
            for row in rows:
                by_n.setdefault(row["n"], []).append(row)
            if by_n:
                ns = sorted(by_n)
                med_wall = [float(np.median([float(r["external_wall_s"]) for r in by_n[n]])) for n in ns]
                q25_wall = [float(np.quantile([float(r["external_wall_s"]) for r in by_n[n]], .25)) for n in ns]
                q75_wall = [float(np.quantile([float(r["external_wall_s"]) for r in by_n[n]], .75)) for n in ns]
                med_sse = [float(np.median([float(r["final_exact_sse"]) for r in by_n[n]])) for n in ns]
                q25_sse = [float(np.quantile([float(r["final_exact_sse"]) for r in by_n[n]], .25)) for n in ns]
                q75_sse = [float(np.quantile([float(r["final_exact_sse"]) for r in by_n[n]], .75)) for n in ns]
                axes[0].errorbar(ns, med_wall, yerr=[np.array(med_wall)-q25_wall, np.array(q75_wall)-med_wall], marker="o", label=method)
                axes[1].errorbar(ns, med_sse, yerr=[np.array(med_sse)-q25_sse, np.array(q75_sse)-med_sse], marker="o", label=method)
        axes[0].set(xscale="log", yscale="log", xlabel="Valid pixels N", ylabel="Complete-fit wall time (s)")
        axes[1].set(xscale="log", yscale="log", xlabel="Valid pixels N", ylabel="Exact final full-data SSE")
        for ax in axes:
            ax.grid(alpha=.25); ax.legend(fontsize=6)
        fig.suptitle(f"31UFS Sentinel-2 workload scaling, K={k}; measured observations and exact targets")
        fig.tight_layout(); fig.savefig(figdir / f"workload_scaling_K{k}.png", dpi=170); plt.close(fig)
    if checkpoints:
        for k in KS:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for method in sorted({r["method"] for r in checkpoints if r["k"] == k}):
                rows = sorted((r for r in checkpoints if r["method"] == method and r["k"] == k),
                              key=lambda r: (r["n"], r["seed"], r["points_processed"]))
                grouped = {}
                for r in rows:
                    grouped.setdefault((r["n"], r["points_processed"]), []).append(float(r["full_exact_sse"]))
                for n in sorted({key[0] for key in grouped}):
                    keys = sorted(key for key in grouped if key[0] == n)
                    x = [key[1] / 1e6 for key in keys]
                    y = [float(np.median(grouped[key])) for key in keys]
                    if x:
                        ax.plot(x, y, marker=".", label=f"{method}, N={n}")
            ax.set(xlabel="Mini-Batch observations processed (millions)", ylabel="Exact full-data SSE",
                   title=f"Common observation checkpoints; K={k}")
            ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
            fig.tight_layout(); fig.savefig(figdir / f"sse_vs_observations_K{k}.png", dpi=170); plt.close(fig)
    for k in KS:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        rows = [r for r in runs if r["k"] == k and r["method"] != "G0"]
        for method in sorted({r["method"] for r in rows}):
            selected = next((r for r in rows if r["method"] == method and r["seed"] == min(x["seed"] for x in rows)), None)
            if not selected:
                continue
            schedule = selected["batch_schedule"]
            schedule = json.loads(schedule) if isinstance(schedule, str) else schedule
            points = [int(x["points_processed"] or 0) / 1e6 for x in schedule if x.get("points_processed") is not None]
            batches = [int(x.get("batch_size", x.get("actual") or x.get("requested") or 0))
                       for x in schedule if x.get("points_processed") is not None]
            if points:
                ax.step(points, batches, where="post", label=f"{method}, N={selected['n']}")
        ax.set(xlabel="Processed observations (millions)", ylabel="Batch size",
               title=f"Mini-Batch schedules (lowest recorded seed), K={k}")
        ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(figdir / f"batch_trajectories_K{k}.png", dpi=170); plt.close(fig)
    for k in KS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for method in sorted({r["method"] for r in runs if r["k"] == k}):
            rows = [r for r in runs if r["method"] == method and r["k"] == k]
            if not rows:
                continue
            ns = sorted({r["n"] for r in rows})
            hit_rates = [float(np.mean([r["target_hit"] for r in rows if r["n"] == n])) for n in ns]
            crossing_times = [float(np.median([r["time_to_quality_s"] for r in rows
                if r["n"] == n and r["time_to_quality_s"] is not None]))
                if any(r["n"] == n and r["time_to_quality_s"] is not None for r in rows) else np.nan
                for n in ns]
            axes[0].plot(ns, hit_rates, marker="o", label=method)
            axes[1].plot(ns, crossing_times, marker="o", label=method)
        axes[0].set(xscale="log", ylim=(-.05, 1.05), xlabel="Valid pixels N", ylabel="Target hit rate")
        axes[1].set(xscale="log", yscale="log", xlabel="Valid pixels N", ylabel="Median observed time to target (s)")
        for ax in axes:
            ax.grid(alpha=.25); ax.legend(fontsize=6)
        fig.suptitle(f"Quality target attainment and time-to-quality; K={k} (misses omitted from time axis)")
        fig.tight_layout(); fig.savefig(figdir / f"time_to_quality_K{k}.png", dpi=170); plt.close(fig)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        output = []
        for row in csv.DictReader(stream):
            for key in ("n", "k", "seed", "repeat", "points_processed", "observations_processed", "updates"):
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            for key in ("external_wall_s", "final_exact_sse", "relative_gap_to_reference", "target_sse"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            if "target_hit" in row:
                row["target_hit"] = row["target_hit"].lower() == "true"
            output.append(row)
        return output


def run(args) -> Path:
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory; existing evidence will not be overwritten: {out}")
    out.mkdir(parents=True)
    for folder in ("phase_cumulative", "phase_decayed", "phase_exact", "figures", "final_summary"):
        (out / folder).mkdir()
    hardware_before = _hardware()
    git = _git()
    if git["branch"] != "experiment-adaptive-v2":
        raise RuntimeError(f"Expected existing experiment-adaptive-v2 branch; found {git['branch']}")
    data = _read_data()
    selected_sizes = tuple(args.sizes) if args.sizes else tuple(data)
    unknown_sizes = sorted(set(selected_sizes) - set(data))
    if unknown_sizes:
        raise ValueError(f"Unsupported dataset sizes requested: {unknown_sizes}")
    if len(set(selected_sizes)) != len(selected_sizes):
        raise ValueError("--sizes must not contain duplicates")
    data = {n: data[n] for n in selected_sizes}
    selected_ks = tuple(args.k_values) if args.k_values else tuple(KS)
    unknown_ks = sorted(set(selected_ks) - set(KS))
    if unknown_ks:
        raise ValueError(f"Unsupported K values requested: {unknown_ks}")
    if len(set(selected_ks)) != len(selected_ks):
        raise ValueError("--k-values must not contain duplicates")
    system_memory_before = _system_memory()
    gpu_memory_estimates = {str(n): _gpu_workspace_estimate(n, 4, max(KS)) for n in data}
    if max(gpu_memory_estimates.values()) > .8 * hardware_before["cuda_memory_free_bytes"]:
        raise MemoryError("Predicted GPU buffer requirement exceeds 80% of currently free VRAM")
    # Numba's own kernels are exercised before measurements. Warm-ups are saved
    # separately and excluded from the result ledger.
    first_X, _ = next(iter(data.values()))
    warmups = _run_warmup(first_X)
    _append_csv(out / "warmups.csv", warmups)

    config = {"study": "real Sentinel-2 workload scalability of exact CUDA and adaptive Mini-Batch",
        "scene": "S2A_MSIL2A_20240625T103631_N0510_R008_T31UFS_20240625T164350.SAFE",
        "split": "development; no test-scene tuning or data used",
        "sizes": {str(n): {"source_dir": str(SIZES[n]), "x_sha256": _sha_bytes(X.tobytes()),
            "metadata_sha256": _sha_file(SIZES[n] / "metadata.json"), "actual_N": len(X),
            "window": scene.get("window"), "standardization_sha256": scene.get("standardization_sha256")}
            for n, (X, scene) in data.items()},
        "D": 4, "K_values": list(selected_ks), "seeds": list(args.seeds), "repeats_per_seed": args.repeats,
        "system_memory_before": system_memory_before,
        "estimated_peak_gpu_workspace_bytes_by_N": gpu_memory_estimates,
        "warmups": "one excluded short GPU fit per method; CUDA JIT warmed before measurements",
        "observation_budget": OBSERVATION_BUDGET, "common_full_data_sse_checkpoints": list(CHECKPOINTS),
        "frozen_quality_rule": "target = 1.05 * exact CUDA G0 converged full-data SSE per N/K/seed; evaluation-only; never passed to Mini-Batch methods/controllers",
        "cumulative_protocol": "shared fixed disjoint 1024-point probe, exact GPU assignment, same cumulative float64 sums/int64 counts; independent per-batch sampling from the probe complement; equal observation budget",
        "decayed_protocol": "same decayed update family, lambda=0.5**(actual_batch_size/(10*N)), exact GPU assignment, fixed probe is diagnostic and overlaps the population; equal observation budget; reported separately from cumulative family",
        "fixed_batch_candidates": list(BATCHES), "adaptive_initial_batch": 8192,
        "adaptive_bounds": [4096, 65536], "sampling_seed": "numpy.default_rng(seed)",
        "sampling_trace_policy": "Cumulative per-update hashing disabled to avoid update-count-dependent timing perturbation; deterministic seed/generator/policy and source hashes provide reproducible sampling provenance. Decayed implementation's built-in hashes remain enabled with sampling_provenance_s recorded. Runtime comparisons remain within update family.",
        "initialization": "same seeded K-Means++ C0 per N/K/seed shared across all methods",
        "assignment": "custom Numba CUDA exact assignment; no CPU fallback or CUDA simulator",
        "timing": "external perf_counter around complete fit call; includes setup/transfer/checkpoint monitoring/final exact assignment; warmups excluded",
        "target_note": "One archived quality-matrix target did not correspond to per-seed exact SSE *1.05 despite its metadata. To retain the frozen 5% criterion without reusing inconsistent numeric thresholds, each new cell uses a separately measured, converged exact CUDA G0 reference objective; controllers never receive it.",
        "protocol_status": "bounded pilot; one or more repetitions per seed as supplied; expansion occurs only for cells with an informative target-attaining or near-target result",
        "time_budget_seconds": args.time_budget_seconds, "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git["branch"], "git_commit": git["commit"], "git_status_before": git["status_porcelain"],
        "source_hashes": {str(p.relative_to(ROOT)): _sha_file(p) for p in [
            Path(__file__), ROOT / "src/pcap/kmeans_cuda.py", ROOT / "src/pcap/minibatch_cuda.py",
            ROOT / "src/pcap/adaptive_cuda.py", ROOT / "src/pcap/budgeted_minibatch.py",
            ROOT / "src/pcap/quality_guard_controller.py", ROOT / "src/pcap/minibatch_decayed_cuda.py",
            ROOT / "src/pcap/adaptive_v2_controller.py", ROOT / "src/pcap/initialization.py"]},
        "packages": _packages(), "platform": platform.platform(), "python": sys.version,
        "hardware_before": hardware_before, "cuda_simulator": False}
    _json(out / "config.json", config)
    _json(out / "manifest.json", {"complete": False, "config": config, "failures": [],
        "source_code_unchanged": True, "artifact_policy": "append per completed fit; raw evidence never overwritten"})

    from pcap.contracts import FitConfig
    from pcap.controlled_fit import fit_controlled
    from pcap.initialization import initialize_centroids
    from pcap.kmeans_cuda import fit_cuda
    # The fit APIs return final exact assignments and SSE. An extra full CPU
    # assignment pass here would perturb execution ordering and is avoided.

    cells = [(n, k, seed) for n in data for k in selected_ks for seed in args.seeds]
    order_rng = np.random.default_rng(3102026)
    cells = [cells[i] for i in order_rng.permutation(len(cells))]
    references: dict[tuple[int, int, int], dict] = {}
    refs = []
    failures = []
    study_start = perf_counter()
    ref_start = study_start
    # Establish an exact, converged G0 objective before any adaptive run in each
    # cell. This is the evaluator reference, not an input to any controller.
    for n, k, seed in cells:
        if perf_counter() - study_start > args.time_budget_seconds:
            failure = {"stage": "G0_reference", "n": n, "k": k, "seed": seed,
                       "error_type": "TimeoutError", "error": "Reference phase time budget exceeded between fits",
                       "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            failures.append(failure)
            _append_csv(out / "failure_records.csv", [failure])
            continue
        X, scene = data[n]
        C0 = initialize_centroids(X, k, seed=seed)
        x_hash, c_hash = _sha_bytes(X.tobytes()), _sha_bytes(C0.tobytes())
        t = perf_counter()
        try:
            reference = fit_cuda(X, C0, FitConfig(k, seed, 1000, 1e-6))
            if reference.metadata.get("termination_reason") != "tolerance":
                raise RuntimeError("exact CUDA reference did not converge within 1000 Lloyd steps")
            target = _target_sse(float(reference.sse))
            item = {"n": n, "k": k, "seed": seed, "X_sha256": x_hash, "C0_sha256": c_hash,
                    "reference_sse": float(reference.sse), "target_sse": target,
                    "reference_converged": True, "reference_steps": int(reference.metadata["steps"]),
                    "reference_wall_s": perf_counter() - t, "source": "new exact CUDA G0 convergence run",
                    "standardization_sha256": scene["standardization_sha256"]}
            references[(n, k, seed)] = item
            refs.append(item)
            _append_csv(out / "references.csv", [item])
        except Exception as exc:
            failure = {"stage": "G0_reference", "n": n, "k": k, "seed": seed,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "wall_s": perf_counter() - t, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            failures.append(failure); _append_csv(out / "failure_records.csv", [failure])
    planned_cells = len(cells)
    cells = [cell for cell in cells if cell in references]

    methods = _method_specs(len(next(iter(data.values()))[0]))
    measured = []
    checkpoint_rows = []
    event_rows = []
    started = study_start
    try:
        for n, k, seed in cells:
            if perf_counter() - started > args.time_budget_seconds:
                raise TimeoutError("Study budget exceeded between complete fit calls")
            X, scene = data[n]
            C0 = initialize_centroids(X, k, seed=seed)
            ref = references[(n, k, seed)]
            for repeat in range(args.repeats):
                case_methods = [dict(method="G0", family="exact", kind="exact_target", batch_size=None)] + methods
                case_methods = [case_methods[i] for i in order_rng.permutation(len(case_methods))]
                for spec in case_methods:
                    if perf_counter() - started > args.time_budget_seconds:
                        raise TimeoutError("Study budget exceeded between complete fit calls")
                    method = spec["method"]
                    t = perf_counter()
                    try:
                        if method == "G0":
                            config_fit = FitConfig(k, seed, 1000, 1e-6)
                            result = fit_controlled(X, C0, config_fit, device="cuda", mode="full",
                                evaluation_interval=1, target_sse=ref["target_sse"])
                            quality = result.metadata.get("quality_history", [])
                            cps = [{"step": q["step"], "elapsed_s": q["elapsed_s"],
                                    "points_processed": q["points_processed"],
                                    "full_exact_sse": q["full_data_sse"],
                                    "target_hit": float(q["full_data_sse"]) <= ref["target_sse"],
                                    "checkpoint_kind": "full_lloyd_evaluation"}
                                   for q in quality]
                            events = []
                        else:
                            result, cps, events = _call_mini(X, C0, seed, k, spec,
                                budget=OBSERVATION_BUDGET, max_steps=10_002)
                            cps = [{"step": q.get("step"), "elapsed_s": q.get("elapsed_s"),
                                    "points_processed": q.get("points_processed"),
                                    "full_exact_sse": q.get("full_exact_sse", q.get("full_data_sse")),
                                    "target_hit": float(q.get("full_exact_sse", q.get("full_data_sse", float("inf")))) <= ref["target_sse"],
                                    "checkpoint_kind": q.get("checkpoint_kind", "fixed_observation_checkpoint")}
                                   for q in cps]
                        external_wall = perf_counter() - t
                        hit = bool(any(float(q["full_exact_sse"]) <= ref["target_sse"] for q in cps))
                        crossing = next((q for q in cps if float(q["full_exact_sse"]) <= ref["target_sse"]), None)
                        md = result.metadata
                        timing = dict(result.timings)
                        row = {"method": method, "family": spec["family"], "kind": spec["kind"],
                            "batch_size_initial": spec["batch_size"], "n": n, "d": 4, "k": k,
                            "seed": seed, "repeat": repeat, "scene_id": scene["scene_id"],
                            "X_sha256": ref["X_sha256"], "C0_sha256": ref["C0_sha256"],
                            "target_sse": ref["target_sse"], "reference_sse": ref["reference_sse"],
                            "final_exact_sse": float(result.sse),
                            "relative_gap_to_reference": (float(result.sse) - ref["reference_sse"]) / ref["reference_sse"],
                            "target_hit": hit,
                            "time_to_quality_s": crossing["elapsed_s"] if crossing else None,
                            "first_target_observations": crossing["points_processed"] if crossing else None,
                            "external_wall_s": external_wall,
                            "observations_processed": int(md.get("points_processed", 0)),
                            "updates": int(md.get("steps", 0)),
                            "termination_reason": md.get("termination_reason"),
                            "convergence_satisfied": bool(md.get("convergence_satisfied", False)),
                            "batch_schedule": _compact_schedule(result.history),
                            "sampling_policy": md.get("sampling_policy"),
                            "sampling_rng": md.get("sampling_rng", "numpy.default_rng(seed)"),
                            "probe_size": md.get("probe_size"), "probe_is_disjoint": md.get("probe_is_disjoint"),
                            "timings": timing, "execution_device": md.get("execution_device"),
                            "gpu": md.get("gpu"), "estimated_device_buffer_bytes": md.get("estimated_device_buffer_bytes"),
                            "quality_checkpoints": len(cps),
                            "final_sse_source": "fit API final exact full-data assignment"}
                        measured.append(row)
                        phase = ("phase_decayed" if spec["family"] == "decayed" else
                                 "phase_cumulative" if spec["family"] == "cumulative" else "phase_exact")
                        _append_csv(out / phase / "individual_runs.csv", [row])
                        new_checkpoints = [{"method": method, "family": spec["family"], "n": n,
                            "d": 4, "k": k, "seed": seed, "repeat": repeat, **q} for q in cps]
                        checkpoint_rows.extend(new_checkpoints)
                        _append_csv(out / "checkpoints.csv", new_checkpoints)
                        for h in result.history:
                            actual_batch = h.get("actual_batch_size", h.get("batch_size"))
                            if actual_batch is None:
                                continue
                            trace = {"method": method, "family": spec["family"], "n": n, "k": k,
                                "seed": seed, "repeat": repeat, "step": h.get("step"),
                                "requested_batch_size": h.get("requested_batch_size", h.get("batch_size")),
                                "actual_batch_size": actual_batch, "points_processed": h.get("points_processed"),
                                "centroid_movement": h.get("centroid_displacement"),
                                "controller_decision": h.get("controller_decision"),
                                "forgetting_factor": h.get("forgetting_factor")}
                            _append_csv(out / "batch_trajectories.csv", [trace])
                            if h.get("sample_indices_sha256"):
                                _append_csv(out / "sampling_provenance.csv", [{**trace,
                                    "sample_indices_sha256": h["sample_indices_sha256"]}])
                        if spec["family"] in ("cumulative", "decayed"):
                            new_events = [{"method": method, "n": n, "k": k, "seed": seed,
                                           "repeat": repeat, **event} for event in (events or [])]
                            new_events += [{"method": method, "n": n, "k": k, "seed": seed,
                                "repeat": repeat, "step": h.get("step"),
                                "points_processed": h.get("points_processed"),
                                "event": "controller_decision", "decision": h.get("controller_decision"),
                                "batch_size": h.get("actual_batch_size", h.get("batch_size")),
                                "next_batch_size": h.get("next_batch_size")}
                                for h in result.history if h.get("controller_decision") is not None]
                            event_rows.extend(new_events)
                            _append_csv(out / "controller_events.csv", new_events)
                        for h in result.history:
                            actual_batch = h.get("actual_batch_size", h.get("batch_size"))
                            if actual_batch is None:
                                continue
                            trace = {"method": method, "family": spec["family"], "n": n, "k": k,
                                "seed": seed, "repeat": repeat, "step": h.get("step"),
                                "requested_batch_size": h.get("requested_batch_size", h.get("batch_size")),
                                "actual_batch_size": actual_batch, "points_processed": h.get("points_processed"),
                                "centroid_movement": h.get("centroid_displacement"),
                                "controller_decision": h.get("controller_decision"),
                                "forgetting_factor": h.get("forgetting_factor")}
                            _append_csv(out / "batch_trajectories.csv", [trace])
                            if h.get("sample_indices_sha256"):
                                _append_csv(out / "sampling_provenance.csv", [{**trace,
                                    "sample_indices_sha256": h["sample_indices_sha256"]}])
                        timing_row = {"method": method, "family": spec["family"], "n": n, "k": k,
                            "seed": seed, "repeat": repeat, "external_wall_s": external_wall,
                            **{f"{key}_s": value for key, value in timing.items()
                               if isinstance(value, (int, float))},
                            "scope_note": "internal components overlap/are subsets per metadata; do not sum to external wall"}
                        _append_csv(out / phase / "timing_breakdown.csv", [timing_row])
                    except Exception as exc:
                        failure = {"stage": "measured_fit", "method": method, "n": n, "k": k,
                            "seed": seed, "repeat": repeat, "error_type": type(exc).__name__,
                            "error": str(exc), "traceback": __import__("traceback").format_exc(),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                        failures.append(failure)
                        _append_csv(out / "failure_records.csv", [failure])
        complete = len(measured) == len(cells) * (1 + len(methods)) * args.repeats and not failures
    except TimeoutError as exc:
        failures.append({"stage": "study_budget", "error_type": "TimeoutError", "error": str(exc),
                         "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        _append_csv(out / "failure_records.csv", failures[-1:])
        complete = False
    except KeyboardInterrupt:
        failure = {"stage": "study_execution", "error_type": "KeyboardInterrupt",
                   "error": "Pilot stopped by operator after discovering a harness checkpoint-field mismatch; raw completed fits were retained",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        failures.append(failure)
        _append_csv(out / "failure_records.csv", [failure])
        complete = False

    _summarize(out, measured, checkpoint_rows)
    hardware_after = _hardware()
    _json(out / "hardware_conditions.json", {"before": hardware_before, "after": hardware_after,
        "system_memory_before": system_memory_before, "system_memory_after": _system_memory()})
    manifest = {"complete": bool(complete), "config": config, "failures": failures,
        "planned_cells": planned_cells, "reference_cells_completed": len(refs),
        "completed_measured_fits": len(measured), "expected_measured_fits": planned_cells * (1 + len(methods)) * args.repeats,
        "reference_fits": len(refs), "pilot_runtime_s": perf_counter() - started,
        "reference_runtime_s": sum(float(r["reference_wall_s"]) for r in refs),
        "hardware_after": hardware_after,
        "interpretation_guardrails": ["No time-to-quality claim for a target miss",
            "Cumulative and decayed update families are separate", "A capped/budget-stopped Mini-Batch run is not convergence",
            "A fixed candidate is called best only within the tested batch-width set and cell"],
        "manifest_written_utc": datetime.now(timezone.utc).isoformat()}
    _json(out / "manifest.json", manifest)
    # The study-index/checksum writer excludes its own checksum file.
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("checksums.sha256", "experiment_index.json"))
    _json(out / "experiment_index.json", {"root": str(out), "complete": bool(complete),
        "files": [{"path": str(p.relative_to(out)), "sha256": _sha_file(p), "bytes": p.stat().st_size}
                  for p in files], "checksums_file": "checksums.sha256"})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    (out / "checksums.sha256").write_text("".join(
        f"{_sha_file(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    print(f"Output: {out}\nCompleted fits: {len(measured)}/{planned_cells*(1+len(methods))*args.repeats}\nComplete: {complete}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", choices=tuple(SIZES), default=None,
                        help="optional dataset-size subset, useful for bounded independent pilot processes")
    parser.add_argument("--k-values", type=int, nargs="+", choices=tuple(KS), default=None,
                        help="optional K subset for a focused confirmatory run")
    parser.add_argument("--time-budget-seconds", type=float, default=3600.0,
                        help="checked between fit calls; running fit calls are not interrupted")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be >= 1")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
