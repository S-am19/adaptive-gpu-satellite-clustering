"""Validate and summarize compatible frozen-target quality study archives.

Each K remains a separate workload. C0/G0 speedups use paired runs only after
the raw records confirm identical X, C0, termination, steps, target and SSE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


_COMMON_CONFIG_KEYS = (
    "scene_dir", "methods", "seeds", "max_steps", "reference_steps",
    "tolerance", "batch_size", "ann_effort", "policy_json", "quality_gap",
    "evaluation_interval", "probe_interval", "warmups", "repeats",
)


def _quartile_iqr(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    if len(values) == 1:
        return 0.0
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return float(q3 - q1)


def _load_study(folder: Path) -> tuple[dict, list[dict], dict]:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError(f"Study is incomplete or failed: {folder}")
    config = manifest.get("config", {})
    if not isinstance(config.get("k"), int):
        raise ValueError(f"Study must have one integer K: {folder}")
    expected = len(config.get("methods", ())) * len(config.get("seeds", ())) * int(config.get("repeats", 0))
    if manifest.get("completed_measured_fits") != expected:
        raise ValueError(f"Measured-fit count does not match config: {folder}")
    if manifest.get("X_sha256") != manifest.get("scene", {}).get("X_sha256"):
        raise ValueError(f"Feature hash differs from scene metadata: {folder}")
    with (folder / "measurements.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"CSV row count does not match config: {folder}")
    keys = [(r["method"], int(r["seed"]), int(r["repeat"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate method/seed/repeat measurement: {folder}")
    raw = {}
    with (folder / "raw.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("phase") != "measured":
                continue
            key = (record["method"], int(record["seed"]), int(record["repeat"]))
            if key in raw:
                raise ValueError(f"Duplicate measured raw record: {folder}, {key}")
            raw[key] = record
    if len(raw) != expected:
        raise ValueError(f"Raw measured-record count does not match config: {folder}")
    return manifest, rows, raw


def summarize_studies(study_dirs: list[Path]) -> tuple[list[dict], list[dict], dict]:
    """Return per-K/method summaries, paired exact speedups, and provenance."""
    if not study_dirs:
        raise ValueError("At least one study directory is required")
    loaded = [(Path(p), *_load_study(Path(p))) for p in study_dirs]
    loaded.sort(key=lambda item: int(item[1]["config"]["k"]))
    ks = [int(item[1]["config"]["k"]) for item in loaded]
    if len(ks) != len(set(ks)):
        raise ValueError("Each input directory must represent a distinct K")

    first = loaded[0][1]
    for folder, manifest, _, _ in loaded[1:]:
        config = manifest["config"]
        if manifest.get("X_sha256") != first.get("X_sha256"):
            raise ValueError("Studies must use identical feature arrays")
        if manifest.get("policy_settings") != first.get("policy_settings"):
            raise ValueError("Studies must use identical frozen policy settings")
        if manifest.get("source_hashes") != first.get("source_hashes"):
            raise ValueError("Studies use different source snapshots")
        if any(config.get(key) != first["config"].get(key) for key in _COMMON_CONFIG_KEYS):
            raise ValueError("Study configurations differ beyond K/output directory")

    summaries = []
    speedups = []
    for folder, manifest, rows, raw in loaded:
        config = manifest["config"]
        k = int(config["k"])
        methods = list(config["methods"])
        for method in methods:
            group = [r for r in rows if r["method"] == method]
            if len(group) != len(config["seeds"]) * int(config["repeats"]):
                raise ValueError(f"Unexpected repeat count for K={k}, method={method}")
            times = [float(r["wall_s"]) for r in group]
            sses = [float(r["final_exact_sse"]) for r in group]
            if not all(math.isfinite(x) and x >= 0 for x in times + sses):
                raise ValueError(f"Non-finite timing or SSE for K={k}, method={method}")
            target_hits = sum(r["target_satisfied"].lower() == "true" for r in group)
            reasons = sorted({r["termination_reason"] for r in group})
            steps = sorted({int(r["steps"]) for r in group})
            summaries.append({
                "k": k, "method": method, "execution_device": group[0]["execution_device"],
                "n": int(manifest["scene"]["selected_pixels"]),
                "d": len(manifest["scene"]["bands"]), "seeds": len(config["seeds"]),
                "repeats_per_seed": int(config["repeats"]), "measured_fits": len(group),
                "target_hits": target_hits, "target_hit_rate": target_hits / len(group),
                "median_final_exact_sse": statistics.median(sses),
                "min_final_exact_sse": min(sses), "max_final_exact_sse": max(sses),
                "median_wall_s": statistics.median(times), "wall_iqr_s": _quartile_iqr(times),
                "steps_observed": ";".join(map(str, steps)),
                "termination_reasons": ";".join(reasons), "study_dir": str(folder),
            })

        if "C0" in methods and "G0" in methods:
            ratios = []
            cpu_rows = { (r["method"], int(r["seed"]), int(r["repeat"])): r for r in rows }
            for seed in config["seeds"]:
                for repeat in range(int(config["repeats"])):
                    cpu = raw[("C0", int(seed), repeat)]
                    gpu = raw[("G0", int(seed), repeat)]
                    if cpu.get("X_sha256") != gpu.get("X_sha256") or cpu.get("C0_sha256") != gpu.get("C0_sha256"):
                        raise ValueError(f"C0/G0 input mismatch at K={k}, seed={seed}, repeat={repeat}")
                    if not math.isclose(float(cpu["target_sse"]), float(gpu["target_sse"]), rel_tol=1e-12, abs_tol=1e-9):
                        raise ValueError(f"C0/G0 target mismatch at K={k}, seed={seed}, repeat={repeat}")
                    cm, gm = cpu["metadata"], gpu["metadata"]
                    for field in ("steps", "termination_reason", "target_satisfied"):
                        if cm.get(field) != gm.get(field):
                            raise ValueError(f"C0/G0 {field} mismatch at K={k}, seed={seed}, repeat={repeat}")
                    if not math.isclose(float(cm["final_exact_sse"]), float(gm["final_exact_sse"]), rel_tol=1e-9, abs_tol=1e-7):
                        raise ValueError(f"C0/G0 final SSE mismatch at K={k}, seed={seed}, repeat={repeat}")
                    cpu_wall = float(cpu_rows[("C0", int(seed), repeat)]["wall_s"])
                    gpu_wall = float(cpu_rows[("G0", int(seed), repeat)]["wall_s"])
                    ratios.append(cpu_wall / gpu_wall)
            speedups.append({
                "k": k, "paired_fits": len(ratios),
                "median_cpu_over_gpu_speedup": statistics.median(ratios),
                "speedup_iqr": _quartile_iqr(ratios),
                "median_cpu_wall_s": statistics.median(float(r["wall_s"]) for r in rows if r["method"] == "C0"),
                "median_gpu_wall_s": statistics.median(float(r["wall_s"]) for r in rows if r["method"] == "G0"),
                "pair_validation": "same X/C0 hashes, steps, termination, target status and final exact SSE",
                "study_dir": str(folder),
            })

    provenance = {
        "feature_sha256": first["X_sha256"],
        "policy_settings": first.get("policy_settings"),
        "source_hashes": first.get("source_hashes"),
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "study_directories": [str(item[0]) for item in loaded],
        "protocol": "Per-K summaries retain separate workloads. Exact C0/G0 ratios use paired runs after raw input/output checks.",
    }
    return summaries, speedups, provenance


def write_summary(study_dirs: list[Path], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    summaries, speedups, provenance = summarize_studies(study_dirs)
    output_dir.mkdir(parents=True)
    with (output_dir / "method_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (output_dir / "exact_speedup.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(speedups[0]))
        writer.writeheader(); writer.writerows(speedups)
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    _plot(summaries, speedups, output_dir)
    return output_dir


def _plot(summaries: list[dict], speedups: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({row["k"] for row in summaries})
    methods = sorted({row["method"] for row in summaries})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, device, title in ((axes[0], "cpu", "CPU methods"), (axes[1], "cuda", "CUDA methods")):
        selected = [m for m in methods if next(r["execution_device"] for r in summaries if r["method"] == m) == device]
        width = 0.8 / max(1, len(selected))
        for i, method in enumerate(selected):
            values = [next(r["median_wall_s"] for r in summaries if r["k"] == k and r["method"] == method) for k in ks]
            x = [j - 0.4 + width / 2 + i * width for j in range(len(ks))]
            ax.bar(x, values, width=width, label=method)
        ax.set(title=title, xlabel="K", ylabel="Median complete fit (s)", xticks=range(len(ks)), xticklabels=ks)
        ax.legend(fontsize=7)
    fig.suptitle("Validation time-to-target runs; device groups shown separately")
    fig.tight_layout(); fig.savefig(output_dir / "median_fit_time_by_k.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.plot(ks, [next(r["median_cpu_over_gpu_speedup"] for r in speedups if r["k"] == k) for k in ks], marker="o")
    ax.set(title="Exact paired C0/G0 validation speedup", xlabel="K", ylabel="Median paired CPU/GPU time ratio", xticks=ks)
    fig.tight_layout(); fig.savefig(output_dir / "exact_speedup_by_k.png", dpi=160); plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_summary(args.studies, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
