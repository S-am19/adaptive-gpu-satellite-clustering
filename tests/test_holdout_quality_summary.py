import csv
import hashlib
import json

import pytest

from experiments.summarize_holdout_quality import summarize


CPU_METHODS = ["C0", "C1"]
CUDA_METHODS = ["G0", "G1"]


def _archive(folder, *, device, methods, seed=0, complete=True, c_hash="c0"):
    folder.mkdir()
    config = {
        "k": 4, "max_steps": 10, "reference_steps": 20, "tolerance": 1e-6,
        "batch_size": 8, "evaluation_interval": 2, "probe_interval": 2,
        "warmups": 1, "repeats": 2, "quality_gap": 0.05,
        "policy_json": "policy.json", "ann_effort": None,
        "budget_seconds": 60.0,
    }
    reference = [{"seed": seed, "C0_sha256": c_hash, "exact_sse": 10.0,
                  "steps": 5, "target_sse": 10.5}]
    manifest = {
        "complete": complete, "failures": [], "completed_measured_fits": len(methods) * 2,
        "config": config, "policy_settings": {"batch": {"minimum": 4}},
        "references": reference, "git_commit": "commit-hash",
        "source_hashes": {"experiments/benchmark_quality.py": "runner-hash", "src/a.py": "a-hash"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    measurements, records = [], []
    for method in methods:
        for repeat in range(2):
            wall = 1.0 + repeat
            hit = method in ("C0", "G0")
            sse = 10.0 if hit else 12.0
            measurement = {
                "method": method, "execution_device": device, "seed": seed,
                "repeat": repeat, "wall_s": wall,
                "time_to_target_s": wall if hit else "",
                "first_observed_target_s": wall if hit else "",
                "target_satisfied": hit, "target_sse": 10.5,
                "final_exact_sse": sse, "steps": 5 if hit else 10,
                "termination_reason": "quality_target" if hit else "max_steps",
                "points_processed": 40, "quality_evaluation_s": 0.1,
            }
            measurements.append(measurement)
            records.append({
                "method": method, "execution_device": device, "phase": "measured",
                "repeat": repeat, "seed": seed, "X_sha256": "x-hash",
                "C0_sha256": c_hash, "wall_s": wall, "target_sse": 10.5,
                "metadata": {"final_exact_sse": sse, "steps": measurement["steps"],
                             "quality_history": [{"elapsed_s": 0.0, "full_data_sse": 15.0},
                                                 {"elapsed_s": wall, "full_data_sse": sse}]},
            })
    with (folder / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader(); writer.writerows(measurements)
    (folder / "raw.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return folder


def _pair(tmp_path, *, cpu_complete=True, cuda_c="c0"):
    cpu = _archive(tmp_path / "cpu", device="cpu", methods=CPU_METHODS,
                   complete=cpu_complete)
    cuda = _archive(tmp_path / "cuda", device="cuda", methods=CUDA_METHODS,
                    c_hash=cuda_c)
    return cpu, cuda


def test_summary_checks_paired_archives_and_writes_separate_device_results(tmp_path):
    cpu, cuda = _pair(tmp_path)
    output = summarize([(cpu, cuda)], tmp_path / "summary")
    with (output / "seed_method_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["execution_group"] for row in rows} == {"cpu", "cuda"}
    c0 = next(row for row in rows if row["method"] == "C0")
    assert c0["target_hits"] == "2"
    assert c0["steps"] == "5"
    assert c0["termination_reasons"] == "quality_target"
    assert (output / "quality_seed0_cpu.png").exists()
    assert (output / "quality_seed0_cuda.png").exists()
    for line in (output / "checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((output / rel).read_bytes()).hexdigest() == digest


def test_summary_rejects_incomplete_archive(tmp_path):
    cpu, cuda = _pair(tmp_path, cpu_complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        summarize([(cpu, cuda)], tmp_path / "summary")


def test_summary_rejects_different_initial_centroids(tmp_path):
    cpu, cuda = _pair(tmp_path, cuda_c="different-c0")
    with pytest.raises(ValueError, match="X or C0 hashes differ"):
        summarize([(cpu, cuda)], tmp_path / "summary")
