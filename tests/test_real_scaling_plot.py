import csv
import json

import pytest

from experiments.plot_real_scaling import summarize_runs


def _run(folder, *, n, window, mutate=None):
    folder.mkdir()
    methods, seeds, repeats = ["C0", "G0"], [0, 1], 2
    config = {
        "methods": methods, "seeds": seeds, "k": 4, "d": 4,
        "batch_size": 4, "max_steps": 1, "tolerance": 1e-6,
        "warmups": 1, "repeats": repeats, "omp_threads": 4,
        "ann_effort": None, "probe_interval": 1, "policy_json": None,
    }
    scene = {
        "scene_id": "scene.SAFE", "selected_pixels": n, "bands": ["B02", "B03", "B04", "B08"],
        "standardization_sha256": "stats-hash", "window": window,
        "X_sha256": f"x-{n}",
    }
    manifest = {"complete": True, "failures": [], "config": config,
                "total_completed": len(methods) * len(seeds) * repeats,
                "scene_metadata": scene}
    (folder / "manifest_1.json").write_text(json.dumps(manifest))
    rows = []
    for seed in seeds:
        for repeat in range(repeats):
            for method, device, wall in (("C0", "cpu_exact", 2.0), ("G0", "cuda_exact", 1.0)):
                sse = 10.0
                if mutate == "sse" and method == "G0" and seed == 1 and repeat == 0:
                    sse = 11.0
                rows.append({
                    "method": method, "implementation": device, "seed": seed,
                    "repeat": repeat, "steps": 1, "termination_reason": "max_steps",
                    "convergence_satisfied": False, "wall_time_s": wall,
                    "final_exact_sse": sse, "X_sha256": scene["X_sha256"],
                    "C0_sha256": f"c-{seed}", "case_id": f"case-{seed}",
                    "d": 4, "k": 4,
                })
    with (folder / "benchmark_results_1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return folder


def test_summarize_runs_checks_matched_work_and_computes_speedup(tmp_path):
    small = _run(tmp_path / "small", n=16, window=[0, 0, 4, 4])
    large = _run(tmp_path / "large", n=64, window=[0, 0, 8, 8])

    summaries, speedups, provenance = summarize_runs([large, small])

    assert [r["n"] for r in speedups] == [16, 64]
    assert all(r["paired_fits"] == 4 for r in speedups)
    assert [r["median_paired_speedup"] for r in speedups] == [2.0, 2.0]
    assert len(summaries) == 4
    assert provenance["method"].startswith("single fixed")


def test_summarize_runs_rejects_cpu_gpu_quality_mismatch(tmp_path):
    small = _run(tmp_path / "small", n=16, window=[0, 0, 4, 4], mutate="sse")
    large = _run(tmp_path / "large", n=64, window=[0, 0, 8, 8])
    with pytest.raises(ValueError, match="C0/G0 output mismatch"):
        summarize_runs([small, large])


def test_summarize_runs_rejects_non_nested_crops(tmp_path):
    small = _run(tmp_path / "small", n=16, window=[0, 0, 4, 4])
    large = _run(tmp_path / "large", n=64, window=[10, 10, 8, 8])
    with pytest.raises(ValueError, match="nested"):
        summarize_runs([small, large])
