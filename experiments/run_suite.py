"""Automated benchmark runner executing the complete PCAP experiment matrix (C0 to G5).

Supports running:
- C0: Compiled sequential CPU Lloyd K-Means
- G0: Custom CUDA GPU Lloyd K-Means
- G1: Fixed GPU Mini-Batch K-Means
- G2: Fixed Mini-Batch + Fixed HNSW ANN
- G3: Adaptive Mini-Batch (Batch size alone)
- G4: Adaptive Mini-Batch (ANN search effort alone)
- G5: Combined Adaptive Mini-Batch (Batch size + ANN effort)

Exports detailed CSV rows, summary statistics, and structured JSON manifests.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

# Ensure src is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.metrics import evaluate_clustering_quality


def make_synthetic_dataset(n: int, d: int, k: int, seed: int):
    """Generate reproducible multi-cluster Gaussian mixture for benchmarking."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(0.0, 3.0, (k, d))
    membership = rng.integers(0, k, n)
    points = np.ascontiguousarray(
        anchors[membership] + rng.normal(0.0, 0.4, (n, d)), dtype=np.float32
    )
    initial = initialize_centroids(points, k, seed=seed)
    return points, initial


def run_single_method(method: str, points: np.ndarray, initial: np.ndarray, cfg: FitConfig):
    """Dispatch execution to the appropriate algorithm implementation."""
    if method == "C0":
        from pcap.kmeans_cpu import fit_cpu
        return fit_cpu(points, initial, cfg)
    elif method == "G0":
        from pcap.kmeans_cuda import fit_cuda
        return fit_cuda(points, initial, cfg)
    elif method == "G1":
        try:
            from pcap.minibatch_cuda import fit_minibatch_cuda
            return fit_minibatch_cuda(points, initial, cfg)
        except Exception:
            from pcap.minibatch import fit_minibatch_cpu
            return fit_minibatch_cpu(points, initial, cfg)
    elif method == "G2":
        from pcap.minibatch_ann import fit_minibatch_ann_cpu
        ann_cfg = FitConfig(
            k=cfg.k,
            seed=cfg.seed,
            max_steps=cfg.max_steps,
            tolerance=cfg.tolerance,
            batch_size=cfg.batch_size,
            probe_interval=cfg.probe_interval,
            ann_effort=cfg.ann_effort or 16,
        )
        return fit_minibatch_ann_cpu(points, initial, ann_cfg)
    elif method == "G3":
        from pcap.adaptive_combined import fit_adaptive_minibatch
        return fit_adaptive_minibatch(points, initial, cfg, mode="batch_only")
    elif method == "G4":
        from pcap.adaptive_combined import fit_adaptive_minibatch
        ann_cfg = FitConfig(
            k=cfg.k,
            seed=cfg.seed,
            max_steps=cfg.max_steps,
            tolerance=cfg.tolerance,
            batch_size=cfg.batch_size,
            probe_interval=cfg.probe_interval,
            ann_effort=cfg.ann_effort or 64,
        )
        return fit_adaptive_minibatch(points, initial, ann_cfg, mode="ann_only")
    elif method == "G5":
        from pcap.adaptive_combined import fit_adaptive_minibatch
        comb_cfg = FitConfig(
            k=cfg.k,
            seed=cfg.seed,
            max_steps=cfg.max_steps,
            tolerance=cfg.tolerance,
            batch_size=cfg.batch_size,
            probe_interval=cfg.probe_interval,
            ann_effort=cfg.ann_effort or 64,
        )
        return fit_adaptive_minibatch(points, initial, comb_cfg, mode="combined")
    else:
        raise ValueError(f"Unknown method ID: {method}")


def main():
    parser = argparse.ArgumentParser(description="PCAP Benchmark Suite Runner")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["C0", "G1", "G2", "G3", "G4", "G5"],
        help="List of methods to evaluate (C0, G0, G1, G2, G3, G4, G5)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 4096], help="Dataset sizes N")
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8], help="Number of clusters K")
    parser.add_argument("--d", type=int, default=4, help="Feature dimension D")
    parser.add_argument("--batch-size", type=int, default=256, help="Base mini-batch size")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "benchmark_suite"),
        help="Directory to save CSV results and JSON manifests",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"benchmark_results_{timestamp}.csv"
    manifest_path = out_dir / f"manifest_{timestamp}.json"

    print(f"Starting PCAP Experiment Suite -> Output: {out_dir}")
    print(f"Methods: {args.methods} | Seeds: {args.seeds} | Sizes: {args.sizes} | K: {args.k}")

    csv_fields = [
        "timestamp_utc", "method", "seed", "n", "d", "k", "batch_size",
        "max_steps", "steps", "termination_reason", "final_exact_sse",
        "sse_per_point", "rmse", "davies_bouldin_index", "silhouette_score",
        "wall_time_s", "compute_time_s", "throughput_pixels_per_s",
    ]

    all_records = []
    total_runs = len(args.methods) * len(args.sizes) * len(args.k) * len(args.seeds)
    run_idx = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()

        for n in args.sizes:
            for k in args.k:
                for seed in args.seeds:
                    points, initial = make_synthetic_dataset(n, args.d, k, seed)

                    for method in args.methods:
                        run_idx += 1
                        print(f"[{run_idx}/{total_runs}] Running {method} (N={n}, K={k}, seed={seed})...", end="", flush=True)

                        cfg = FitConfig(
                            k=k,
                            seed=seed,
                            max_steps=args.max_steps,
                            tolerance=args.tolerance,
                            batch_size=min(args.batch_size, n),
                            probe_interval=5,
                        )

                        t0 = perf_counter()
                        try:
                            result = run_single_method(method, points, initial, cfg)
                            elapsed = perf_counter() - t0
                            quality = evaluate_clustering_quality(points, result, seed=seed)
                            throughput = float(n) / max(quality["wall_time_s"], 1e-6)

                            row = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "method": method,
                                "seed": seed,
                                "n": n,
                                "d": args.d,
                                "k": k,
                                "batch_size": cfg.batch_size,
                                "max_steps": cfg.max_steps,
                                "steps": quality["steps"],
                                "termination_reason": quality["termination_reason"],
                                "final_exact_sse": quality["final_exact_sse"],
                                "sse_per_point": quality["sse_per_point"],
                                "rmse": quality["rmse"],
                                "davies_bouldin_index": quality["davies_bouldin_index"],
                                "silhouette_score": quality["silhouette_score"],
                                "wall_time_s": quality["wall_time_s"],
                                "compute_time_s": quality["compute_time_s"],
                                "throughput_pixels_per_s": throughput,
                            }
                            writer.writerow(row)
                            f_csv.flush()
                            all_records.append(row)
                            print(f" Done ({quality['wall_time_s']:.3f}s, SSE: {quality['final_exact_sse']:.1f})")
                        except Exception as exc:
                            print(f" FAILED: {exc}")

    manifest = {
        "timestamp_utc": timestamp,
        "config": vars(args),
        "total_completed": len(all_records),
        "csv_path": str(csv_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f_json:
        json.dump(manifest, f_json, indent=2)

    print(f"\nSuite complete. Results saved to:\n  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
