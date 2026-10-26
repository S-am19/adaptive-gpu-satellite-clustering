import csv
import json

import pytest

from experiments.summarize_quality_matrix import summarize_studies


def _write_study(folder, *, k=4, x_hash="x-hash", c_hash="c-hash", complete=True):
    folder.mkdir()
    seeds = [0, 1]
    repeats = 2
    methods = ["C0", "G0"]
    config = {
        "scene_dir": "validation512", "output_dir": str(folder), "methods": methods,
        "seeds": seeds, "k": k, "max_steps": 10, "reference_steps": 20,
        "tolerance": 1e-6, "batch_size": 8, "ann_effort": None,
        "policy_json": "policy.json", "quality_gap": 0.05,
        "evaluation_interval": 2, "probe_interval": 2, "warmups": 1,
        "repeats": repeats, "budget_seconds": 60.0,
    }
    manifest = {
        "complete": complete, "failures": [],
        "completed_measured_fits": len(methods) * len(seeds) * repeats,
        "config": config, "X_sha256": x_hash,
        "scene": {"X_sha256": x_hash, "selected_pixels": 16, "bands": ["B2", "B3", "B4", "B8"]},
        "policy_settings": {"batch": {"minimum": 4}}, "source_hashes": {"src/a.py": "source-hash"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    raw = []
    for seed in seeds:
        for repeat in range(repeats):
            for method, device, wall in (("C0", "cpu", 2.0), ("G0", "cuda", 1.0)):
                rows.append({
                    "method": method, "execution_device": device, "seed": seed,
                    "repeat": repeat, "wall_s": wall, "target_satisfied": "True",
                    "target_sse": 10.5, "final_exact_sse": 10.0,
                    "steps": 5, "termination_reason": "quality_target",
                })
                raw.append({
                    "method": method, "phase": "measured", "seed": seed,
                    "repeat": repeat, "X_sha256": x_hash, "C0_sha256": c_hash,
                    "target_sse": 10.5,
                    "metadata": {
                        "steps": 5, "termination_reason": "quality_target",
                        "target_satisfied": True, "final_exact_sse": 10.0,
                    },
                })
    with (folder / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (folder / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    return folder


def test_summary_validates_paired_exact_work_and_reports_medians(tmp_path):
    study = _write_study(tmp_path / "k4")
    summaries, speedups, provenance = summarize_studies([study])

    assert len(summaries) == 2
    assert speedups[0]["k"] == 4
    assert speedups[0]["paired_fits"] == 4
    assert speedups[0]["median_cpu_over_gpu_speedup"] == 2.0
    assert speedups[0]["speedup_iqr"] == 0.0
    assert provenance["feature_sha256"] == "x-hash"


def test_summary_rejects_incomplete_study(tmp_path):
    study = _write_study(tmp_path / "partial", complete=False)
    with pytest.raises(ValueError, match="incomplete or failed"):
        summarize_studies([study])


def test_summary_rejects_feature_mismatch_across_k(tmp_path):
    k4 = _write_study(tmp_path / "k4", k=4, x_hash="x-hash")
    k8 = _write_study(tmp_path / "k8", k=8, x_hash="other-hash")
    with pytest.raises(ValueError, match="identical feature arrays"):
        summarize_studies([k4, k8])


def test_summary_rejects_cpu_gpu_initialization_mismatch(tmp_path):
    study = _write_study(tmp_path / "k4")
    raw_path = study / "raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    next(r for r in records if r["method"] == "G0")["C0_sha256"] = "different-c0"
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="C0/G0 input mismatch"):
        summarize_studies([study])
