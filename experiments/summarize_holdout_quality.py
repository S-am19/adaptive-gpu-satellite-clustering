"""Validate and summarize separate CPU/CUDA held-out quality archives.

Each ``--seed-pair`` contains one completed CPU group and one completed CUDA
group for the same seed. Timing distributions remain separated by device;
this report intentionally does not calculate cross-device speedups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_group(folder: Path, expected_device: str):
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError(f"{folder} is incomplete or has recorded failures")
    rows = list(csv.DictReader((folder / "measurements.csv").open(newline="")))
    raw = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
    measured = [record for record in raw if record.get("phase") == "measured"]
    config = manifest["config"]
    expected_repeats = int(config["repeats"])
    methods = sorted({record["method"] for record in measured})
    if not methods or len(rows) != len(measured):
        raise ValueError(f"{folder} has no methods or CSV/raw row counts differ")
    if manifest.get("completed_measured_fits") != len(measured):
        raise ValueError(f"{folder} completed-fit count does not match its raw records")
    seeds = {int(record["seed"]) for record in measured}
    devices = {record["execution_device"] for record in measured}
    if len(seeds) != 1 or devices != {expected_device}:
        raise ValueError(f"{folder} must contain one seed on {expected_device}")
    seed = next(iter(seeds))
    expected_keys = {(method, repeat) for method in methods for repeat in range(expected_repeats)}
    csv_keys = {(row["method"], int(row["repeat"])) for row in rows}
    raw_keys = {(record["method"], int(record["repeat"])) for record in measured}
    if csv_keys != expected_keys or raw_keys != expected_keys:
        raise ValueError(f"{folder} does not contain every measured method/repetition")
    if any(int(record["metadata"]["steps"]) > int(config["max_steps"]) for record in measured):
        raise ValueError(f"{folder} records a fit beyond max_steps")
    x_hashes = {record["X_sha256"] for record in measured}
    c_hashes = {record["C0_sha256"] for record in measured}
    if len(x_hashes) != 1 or len(c_hashes) != 1:
        raise ValueError(f"{folder} changed X or C0 within a group")
    raw_by_key = {(record["method"], int(record["repeat"])): record for record in measured}
    for row in rows:
        record = raw_by_key[(row["method"], int(row["repeat"]))]
        if record["execution_device"] != expected_device:
            raise ValueError(f"{folder} has a method on the wrong execution device")
        if not np.isclose(float(row["wall_s"]), float(record["wall_s"]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{folder} CSV/raw timing mismatch")
        if not np.isclose(float(row["final_exact_sse"]), float(record["metadata"]["final_exact_sse"]),
                          rtol=1e-10, atol=1e-8):
            raise ValueError(f"{folder} CSV/raw final SSE mismatch")
        if int(row["steps"]) != int(record["metadata"]["steps"]):
            raise ValueError(f"{folder} CSV/raw step count mismatch")
    return {
        "folder": folder,
        "manifest": manifest,
        "config": config,
        "rows": rows,
        "raw": raw,
        "measured": measured,
        "methods": methods,
        "seed": seed,
        "X_sha256": next(iter(x_hashes)),
        "C0_sha256": next(iter(c_hashes)),
        "device": expected_device,
    }


def _same_protocol(a, b):
    keys = (
        "k", "max_steps", "reference_steps", "tolerance", "batch_size",
        "evaluation_interval", "probe_interval", "warmups", "repeats",
        "quality_gap", "ann_effort", "policy_json", "budget_seconds",
    )
    for key in keys:
        if a["config"].get(key) != b["config"].get(key):
            raise ValueError(f"CPU/CUDA protocol mismatch for {key}")
    if a["manifest"].get("policy_settings") != b["manifest"].get("policy_settings"):
        raise ValueError("CPU/CUDA frozen policy settings differ")
    if a["X_sha256"] != b["X_sha256"] or a["C0_sha256"] != b["C0_sha256"]:
        raise ValueError("CPU/CUDA X or C0 hashes differ")
    if a["manifest"].get("references") != b["manifest"].get("references"):
        raise ValueError("CPU/CUDA exact references differ")
    if a["manifest"].get("git_commit") != b["manifest"].get("git_commit"):
        raise ValueError("CPU/CUDA code commits differ")
    source_a = a["manifest"].get("source_hashes", {})
    source_b = b["manifest"].get("source_hashes", {})
    common = set(source_a) & set(source_b)
    if any(source_a[name] != source_b[name] for name in common):
        raise ValueError("CPU/CUDA source snapshots differ")
    runner = "experiments/benchmark_quality.py"
    if source_a.get(runner) != source_b.get(runner):
        raise ValueError("CPU/CUDA benchmark runner snapshots differ")


def _summary_rows(group):
    output = []
    for method in group["methods"]:
        rows = [row for row in group["rows"] if row["method"] == method]
        times = np.asarray([float(row["wall_s"]) for row in rows], dtype=np.float64)
        sses = np.asarray([float(row["final_exact_sse"]) for row in rows], dtype=np.float64)
        output.append({
            "seed": group["seed"],
            "execution_group": group["device"],
            "method": method,
            "repeats": len(rows),
            "target_hits": sum(row["target_satisfied"].lower() == "true" for row in rows),
            "median_wall_s": float(np.median(times)),
            "q1_wall_s": float(np.quantile(times, 0.25)),
            "q3_wall_s": float(np.quantile(times, 0.75)),
            "iqr_wall_s": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
            "median_final_exact_sse": float(np.median(sses)),
            "steps": ";".join(map(str, sorted({int(row["steps"]) for row in rows}))),
            "termination_reasons": ";".join(sorted({row["termination_reason"] for row in rows})),
            "target_sse": float(rows[0]["target_sse"]),
            "X_sha256": group["X_sha256"],
            "C0_sha256": group["C0_sha256"],
        })
    return output


def summarize(seed_pairs, output_dir: Path):
    """Validate seed-paired archives and write CSV, trajectory plots and hashes."""
    if not seed_pairs:
        raise ValueError("At least one CPU/CUDA seed pair is required")
    pairs = []
    for cpu_path, cuda_path in seed_pairs:
        cpu = _read_group(Path(cpu_path), "cpu")
        cuda = _read_group(Path(cuda_path), "cuda")
        if cpu["seed"] != cuda["seed"]:
            raise ValueError("CPU/CUDA seed values differ")
        _same_protocol(cpu, cuda)
        pairs.append((cpu, cuda))
    pairs.sort(key=lambda pair: pair[0]["seed"])
    if len({pair[0]["seed"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate seed pair")
    reference = pairs[0][0]
    for cpu, cuda in pairs:
        for group in (cpu, cuda):
            if group["X_sha256"] != reference["X_sha256"]:
                raise ValueError("Feature matrix differs across seeds")
            reference_group = next(g for g in pairs[0] if g["device"] == group["device"])
            if group["methods"] != reference_group["methods"]:
                raise ValueError("CPU method list differs across seeds")
            if group["manifest"].get("policy_settings") != reference["manifest"].get("policy_settings"):
                raise ValueError("Frozen policy differs across seeds")
            if group["manifest"].get("git_commit") != reference["manifest"].get("git_commit"):
                raise ValueError("Source commits differ across seeds")
            original_sources = reference_group["manifest"].get("source_hashes", {})
            current_sources = group["manifest"].get("source_hashes", {})
            common_sources = set(original_sources) & set(current_sources)
            if any(original_sources[name] != current_sources[name] for name in common_sources):
                raise ValueError("Source snapshot differs across seeds")
            runner = "experiments/benchmark_quality.py"
            if original_sources.get(runner) != current_sources.get(runner):
                raise ValueError("Benchmark runner differs across seeds")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    per_seed = []
    all_groups = []
    for cpu, cuda in pairs:
        for group in (cpu, cuda):
            all_groups.append(group)
            per_seed.extend(_summary_rows(group))
    _write_csv(output_dir / "seed_method_summary.csv", per_seed)

    aggregate = []
    for device in ("cpu", "cuda"):
        methods = sorted({row["method"] for row in per_seed if row["execution_group"] == device})
        for method in methods:
            rows = [row for row in per_seed if row["execution_group"] == device and row["method"] == method]
            times = np.asarray([
                float(row["wall_s"])
                for group in all_groups if group["device"] == device
                for row in group["rows"] if row["method"] == method
            ], dtype=np.float64)
            sses = np.asarray([row["median_final_exact_sse"] for row in rows], dtype=np.float64)
            aggregate.append({
                "execution_group": device,
                "method": method,
                "seeds": len(pairs),
                "measured_repeats": int(len(times)),
                "target_hits": sum(row["target_hits"] for row in rows),
                "median_wall_s": float(np.median(times)),
                "q1_wall_s": float(np.quantile(times, 0.25)),
                "q3_wall_s": float(np.quantile(times, 0.75)),
                "iqr_wall_s": float(np.quantile(times, 0.75) - np.quantile(times, 0.25)),
                "median_final_exact_sse": float(np.median(sses)),
                "timing_interpretation": "within device group only; do not form cross-device speedups",
            })
    _write_csv(output_dir / "aggregate_method_summary.csv", aggregate)
    _plot_trajectories(pairs, output_dir)

    manifest = {
        "purpose": "Validated summary of frozen held-out quality archives; no new fits executed.",
        "seeds": [cpu["seed"] for cpu, _ in pairs],
        "X_sha256": reference["X_sha256"],
        "config": reference["config"],
        "policy_settings": reference["manifest"].get("policy_settings"),
        "git_commit": reference["manifest"].get("git_commit"),
        "source_hashes": reference["manifest"].get("source_hashes"),
        "archives": [
            {"seed": cpu["seed"], "cpu": str(cpu["folder"]), "cuda": str(cuda["folder"]),
             "C0_sha256": cpu["C0_sha256"], "reference": cpu["manifest"].get("references")}
            for cpu, cuda in pairs
        ],
        "timing_note": "CPU and CUDA groups were run separately. Summaries intentionally omit cross-device speedups.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    _write_checksums(output_dir)
    return output_dir


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty summary")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trajectories(pairs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cpu, cuda in pairs:
        for group in (cpu, cuda):
            fig, ax = plt.subplots(figsize=(9, 5))
            for method in group["methods"]:
                candidates = [
                    record for record in group["raw"]
                    if record.get("phase") == "measured"
                    and record.get("repeat") == 0 and record.get("method") == method
                ]
                if not candidates:
                    continue
                history = candidates[0]["metadata"].get("quality_history", [])
                ax.plot([point["elapsed_s"] for point in history],
                        [point["full_data_sse"] for point in history],
                        marker=".", label=method)
            seed = group["seed"]
            reference = next(item for item in group["manifest"]["references"] if item["seed"] == seed)
            ax.axhline(reference["target_sse"], color="black", linestyle="--", label="Frozen target")
            ax.set(xlabel="Instrumented complete-fit elapsed seconds", ylabel="Exact full-data SSE",
                   title=f"Held-out seed {seed}, {group['device']} group; measured repeat 0")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(output_dir / f"quality_seed{seed}_{group['device']}.png", dpi=140)
            plt.close(fig)


def _write_checksums(folder):
    files = sorted(path for path in folder.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (folder / "checksums.sha256").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(folder).as_posix()}\n" for path in files)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pair", nargs=2, action="append", required=True,
                        metavar=("CPU_ARCHIVE", "CUDA_ARCHIVE"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize(args.seed_pair, args.output_dir)
    print(f"Validated summary written to {args.output_dir}")


if __name__ == "__main__":
    main()
