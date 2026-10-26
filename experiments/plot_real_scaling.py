"""Summarize compatible real-scene fixed-work runs and plot N scaling.

This tool accepts only completed one-update studies with identical K, methods,
seeds, repeat count and frozen settings. Exact C0/G0 ratios are paired and
validated per seed/repeat. Mini-Batch ratios are not reported as GPU speedups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


_CONFIG_KEYS = ("methods", "seeds", "k", "d", "batch_size", "max_steps",
                "tolerance", "warmups", "repeats", "omp_threads",
                "ann_effort", "probe_interval", "policy_json")


def _iqr(values):
    if len(values) == 1:
        return 0.0
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return float(q3 - q1)


def _read_run(folder):
    folder = Path(folder)
    manifests = sorted(folder.glob("manifest_*.json"))
    csvs = sorted(folder.glob("benchmark_results_*.csv"))
    if len(manifests) != 1 or len(csvs) != 1:
        raise ValueError(f"Expected one manifest and one result CSV in {folder}")
    manifest = json.loads(manifests[0].read_text())
    if not manifest.get("complete") or manifest.get("failures"):
        raise ValueError(f"Run is incomplete or failed: {folder}")
    config = manifest.get("config", {})
    if config.get("max_steps") != 1:
        raise ValueError(f"Only one-update fixed-work runs are supported: {folder}")
    with csvs[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(config["methods"]) * len(config["seeds"]) * int(config["repeats"])
    if manifest.get("total_completed") != expected or len(rows) != expected:
        raise ValueError(f"Measured row count does not match config: {folder}")
    if not rows or any(int(r["steps"]) != 1 or r["termination_reason"] != "max_steps" for r in rows):
        raise ValueError(f"A fit did not perform exactly one capped update: {folder}")
    if any(r["convergence_satisfied"].lower() != "false" for r in rows):
        raise ValueError(f"One-update run was incorrectly marked converged: {folder}")
    if len({(int(r["k"]), int(r["d"])) for r in rows}) != 1:
        raise ValueError(f"Run CSV contains multiple K/D cases: {folder}")
    keys = [(r["method"], int(r["seed"]), int(r["repeat"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate method/seed/repeat rows: {folder}")
    return folder, manifest, rows


def summarize_runs(run_dirs):
    """Return per-size method summaries and paired exact CPU/GPU speedups."""
    runs = sorted((_read_run(p) for p in run_dirs), key=lambda x: int(x[1]["scene_metadata"]["selected_pixels"]))
    if len(runs) < 2:
        raise ValueError("At least two real-scene sizes are required")
    sizes = [int(m["scene_metadata"]["selected_pixels"]) for _, m, _ in runs]
    if len(sizes) != len(set(sizes)):
        raise ValueError("Input runs must have distinct valid-pixel counts")
    base = runs[0][1]
    base_scene = base["scene_metadata"]
    for _, manifest, _ in runs[1:]:
        config = manifest["config"]
        if any(config.get(key) != base["config"].get(key) for key in _CONFIG_KEYS):
            raise ValueError("Run configurations differ beyond scene size/output directory")
        scene = manifest["scene_metadata"]
        if scene.get("scene_id") != base_scene.get("scene_id"):
            raise ValueError("Runs must come from the same source scene")
        if scene.get("standardization_sha256") != base_scene.get("standardization_sha256"):
            raise ValueError("Runs must use identical frozen training standardization")
        if scene.get("bands") != base_scene.get("bands"):
            raise ValueError("Runs must use the same feature bands")
    # Require spatially nested crops so changing N does not switch source scenes.
    for i, (_, mi, _) in enumerate(runs):
        wi = mi["scene_metadata"]["window"]
        for _, mj, _ in runs[i + 1:]:
            wj = mj["scene_metadata"]["window"]
            small, large = (wi, wj) if wi[2] * wi[3] < wj[2] * wj[3] else (wj, wi)
            if not (large[0] <= small[0] and large[1] <= small[1]
                    and large[0] + large[2] >= small[0] + small[2]
                    and large[1] + large[3] >= small[1] + small[3]):
                raise ValueError("Real-scene crop windows must be nested")

    summaries, speedups = [], []
    for folder, manifest, rows in runs:
        n = int(manifest["scene_metadata"]["selected_pixels"])
        if any(r["X_sha256"] != manifest["scene_metadata"]["X_sha256"] for r in rows):
            raise ValueError(f"Run rows do not match prepared-scene hash: {folder}")
        for method in manifest["config"]["methods"]:
            group = [r for r in rows if r["method"] == method]
            expected = len(manifest["config"]["seeds"]) * int(manifest["config"]["repeats"])
            if len(group) != expected:
                raise ValueError(f"Unexpected repeats for N={n}, method={method}")
            times = [float(r["wall_time_s"]) for r in group]
            sses = [float(r["final_exact_sse"]) for r in group]
            if not all(math.isfinite(x) and x >= 0 for x in times + sses):
                raise ValueError("Non-finite timing or SSE encountered")
            q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive") if len(times) > 1 else (times[0], times[0], times[0])
            summaries.append({
                "n": n, "d": int(group[0]["d"]), "k": int(group[0]["k"]),
                "method": method, "implementation": group[0]["implementation"],
                "measured_fits": len(group), "median_wall_s": statistics.median(times),
                "wall_iqr_s": _iqr(times), "q25_wall_s": q1, "q75_wall_s": q3,
                "median_final_exact_sse": statistics.median(sses),
                "min_final_exact_sse": min(sses), "max_final_exact_sse": max(sses),
                "termination_reason": "max_steps", "steps": 1, "scene_dir": str(folder),
            })
        if "C0" in manifest["config"]["methods"] and "G0" in manifest["config"]["methods"]:
            lookup = {(r["method"], int(r["seed"]), int(r["repeat"])): r for r in rows}
            ratios = []
            for seed in manifest["config"]["seeds"]:
                for repeat in range(int(manifest["config"]["repeats"])):
                    cpu, gpu = lookup[("C0", int(seed), repeat)], lookup[("G0", int(seed), repeat)]
                    if cpu["X_sha256"] != gpu["X_sha256"] or cpu["C0_sha256"] != gpu["C0_sha256"] or cpu["case_id"] != gpu["case_id"]:
                        raise ValueError(f"C0/G0 input mismatch at N={n}, seed={seed}, repeat={repeat}")
                    if (cpu["steps"] != gpu["steps"] or cpu["termination_reason"] != gpu["termination_reason"]
                            or not math.isclose(float(cpu["final_exact_sse"]), float(gpu["final_exact_sse"]), rel_tol=1e-5, abs_tol=1e-7)):
                        raise ValueError(f"C0/G0 output mismatch at N={n}, seed={seed}, repeat={repeat}")
                    ratios.append(float(cpu["wall_time_s"]) / float(gpu["wall_time_s"]))
            q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive") if len(ratios) > 1 else (ratios[0], ratios[0], ratios[0])
            speedups.append({
                "n": n,
                "d": int(lookup[("C0", int(manifest["config"]["seeds"][0]), 0)]["d"]),
                "k": int(lookup[("C0", int(manifest["config"]["seeds"][0]), 0)]["k"]),
                "paired_fits": len(ratios), "median_paired_speedup": statistics.median(ratios),
                "speedup_iqr": _iqr(ratios), "q25_speedup": q1, "q75_speedup": q3,
                "median_cpu_wall_s": statistics.median(float(r["wall_time_s"]) for r in rows if r["method"] == "C0"),
                "median_gpu_wall_s": statistics.median(float(r["wall_time_s"]) for r in rows if r["method"] == "G0"),
                "scope": "C0/G0 paired one-update complete fit; includes transfer/final exact assignment",
            })
    provenance = {
        "study_directories": [str(folder) for folder, _, _ in runs],
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_scene_id": base_scene.get("scene_id"),
        "feature_bands": base_scene.get("bands"),
        "standardization_sha256": base_scene.get("standardization_sha256"),
        "method": "single fixed Lloyd/Mini-Batch update; not convergence or time-to-quality",
        "timing_scope": "external synchronized complete fit; saved per-method summaries use median and IQR over seeds/repeats",
        "interpretation": "C0/G0 ratios are matched exact-work observations. No CPU/GPU speedup inferred for methods processing different work.",
    }
    return summaries, speedups, provenance


def write_report(run_dirs, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    summaries, speedups, provenance = summarize_runs(run_dirs)
    output_dir.mkdir(parents=True)
    for filename, rows in (("method_summary.csv", summaries), ("exact_speedup.csv", speedups)):
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    _plots(summaries, speedups, output_dir)
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    (output_dir / "checksums.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
    return output_dir


def _plots(summaries, speedups, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted({r["method"] for r in summaries})
    ns = sorted({r["n"] for r in summaries})
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        rows = sorted((r for r in summaries if r["method"] == method), key=lambda r: r["n"])
        ax.errorbar([r["n"] for r in rows], [r["median_wall_s"] for r in rows],
                    yerr=[[r["median_wall_s"] - r["q25_wall_s"] for r in rows],
                          [r["q75_wall_s"] - r["median_wall_s"] for r in rows]],
                    marker="o", label=method)
    ax.set(xscale="log", yscale="log", xlabel="Valid Sentinel-2 pixels (N)",
           ylabel="Median external one-update fit time (s)", title="Real-scene fixed-work scaling; one session per crop")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output_dir / "real_scaling_median_iqr.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.errorbar([r["n"] for r in speedups], [r["median_paired_speedup"] for r in speedups],
                yerr=[[r["median_paired_speedup"] - r["q25_speedup"] for r in speedups],
                      [r["q75_speedup"] - r["median_paired_speedup"] for r in speedups]],
                marker="o")
    ax.set(xscale="log", xlabel="Valid Sentinel-2 pixels (N)", ylabel="Median paired C0/G0 time ratio",
           title="Exact CPU/GPU one-update comparison; >1 means GPU faster")
    fig.tight_layout(); fig.savefig(output_dir / "exact_cpu_gpu_speedup.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_report(args.runs, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
