"""CPU-only checks for benchmark reproducibility and reporting correctness."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig, RunResult
from pcap.kmeans_cpu import fit_cpu


spec = importlib.util.spec_from_file_location(
    "benchmark_cuda", Path(__file__).resolve().parents[1] / "experiments/benchmark_cuda.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_synthetic_reproducibility():
    first = benchmark.make_synthetic(100, 14, 4, 17)
    second = benchmark.make_synthetic(100, 14, 4, 17)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32 and first.flags.c_contiguous
    assert benchmark.array_hash(first) == benchmark.array_hash(second)
    assert benchmark.array_hash(first) != benchmark.array_hash(benchmark.make_synthetic(100, 14, 4, 18))


@pytest.mark.parametrize("args", [["--repeats", "4"], ["--warmups", "0"],
                                 ["--seeds", "-1"], ["--tolerance", "nan"],
                                 ["--sizes", "2", "--clusters", "4"]])
def test_reject_invalid_benchmark_settings(args):
    with pytest.raises(SystemExit):
        benchmark.parse_args(args)


def test_summary_excludes_warmup_and_reports_spread():
    rows = []
    for phase in ("warmup", "measured"):
        for method, factor in (("cpu", 2.0), ("gpu", 1.0)):
            for repeat in range(5):
                row = dict(case_id="test", n=100, d=4, k=2, seed=0, phase=phase,
                           method=method, external_fit_s=(1000 if phase == "warmup" else factor * (repeat + 1)),
                           quality_valid=True, steps=3, termination_reason="tolerance",
                           convergence_satisfied=True, final_centroid_displacement=0.0,
                           quality_json=json.dumps({"relative_sse_gap": 0.0}))
                row.update({key: 0.1 for key in benchmark.TIMING_KEYS})
                rows.append(row)
    result = benchmark.summarize(rows)
    assert result["repeats"] == 5
    assert result["cpu_median_s"] == 6.0
    assert result["gpu_median_s"] == 3.0
    assert result["cpu_iqr_s"] == 4.0
    assert result["end_to_end_speedup"] == 2.0
    assert json.loads(result["cpu_iteration_counts"]) == [3] * 5
    assert json.loads(result["gpu_termination_reasons"]) == ["tolerance"]
    assert result["gpu_convergence_satisfied"] is True
    assert result["cpu_converged_repeats"] == 5
    assert result["cpu_final_displacement_max"] == 0.0
    # Matching limit exits can be valid comparisons without convergence.
    for row in rows:
        if row["phase"] == "measured":
            row.update(steps=30, termination_reason="max_steps",
                       convergence_satisfied=False, final_centroid_displacement=0.25)
    limit = benchmark.summarize(rows)
    assert limit["quality_valid"] is True and limit["end_to_end_speedup"] == 2.0
    assert json.loads(limit["cpu_iteration_counts"]) == [30] * 5
    assert json.loads(limit["cpu_termination_reasons"]) == ["max_steps"]
    assert limit["cpu_convergence_satisfied"] is False
    assert limit["gpu_converged_repeats"] == 0
    assert limit["gpu_final_displacement_max"] == 0.25
    rows[-1]["quality_valid"] = False
    assert benchmark.summarize(rows)["end_to_end_speedup"] == ""


def test_quality_gate_checks_final_sse_and_equal_work(two_groups):
    points, initial = two_groups
    cpu = fit_cpu(points, initial, FitConfig(2, 0, 10, 0.0))
    other = RunResult(cpu.centroids.copy(), cpu.labels.copy(), [], {}, dict(cpu.metadata))
    valid, checks = benchmark.verify_quality(points, cpu, other, assign_exact)
    assert valid and checks["label_agreement"] == 1.0
    other.metadata["steps"] += 1
    valid, checks = benchmark.verify_quality(points, cpu, other, assign_exact)
    assert not valid and not checks["steps_agree"]
    other.metadata["steps"] -= 1
    other.metadata["final_exact_sse"] += 10
    valid, checks = benchmark.verify_quality(points, cpu, other, assign_exact)
    assert not valid and not checks["gpu_sse_exact"]


@pytest.mark.parametrize("tolerance,expected_reason,satisfied", [
    (0.0, "max_steps", False), (1.0, "tolerance", True), (2.0, "tolerance", True),
])
def test_convergence_at_step_limit_uses_actual_movement(two_groups, tolerance, expected_reason, satisfied):
    result = fit_cpu(*two_groups, FitConfig(2, 0, 1, tolerance))
    fields = benchmark._convergence_fields(result, tolerance)
    assert result.metadata["steps"] == 1
    assert result.metadata["termination_reason"] == expected_reason
    assert fields == {"final_centroid_displacement": 1.0, "convergence_satisfied": satisfied}


def test_reject_inconsistent_convergence_metadata(two_groups):
    result = fit_cpu(*two_groups, FitConfig(2, 0, 1, 0.0))
    result.metadata["termination_reason"] = "tolerance"
    with pytest.raises(ValueError, match="Termination reason"):
        benchmark._convergence_fields(result, 0.0)
    result.history[-1]["centroid_displacement"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        benchmark._convergence_fields(result, 0.0)


@pytest.mark.parametrize("enabled", [False, True])
def test_gc_policy_reports_actual_state_without_changing_it(monkeypatch, enabled):
    monkeypatch.setattr(benchmark.gc, "isenabled", lambda: enabled)
    policy = benchmark._timing_policy_metadata()
    assert "explicit gc.collect()" in policy["excluded"]
    assert ", GC," not in policy["excluded"]
    gc_policy = policy["garbage_collection"]
    assert gc_policy["explicit_collect_outside_timer"] is True
    assert gc_policy["automatic_gc_enabled_at_start"] is enabled
    assert "inside measured fit calls" in gc_policy["automatic_gc_policy"]
    assert benchmark.gc.isenabled() is enabled


def test_summary_reports_mixed_convergence_and_excludes_warmups():
    rows = []
    for method in ("cpu", "gpu"):
        for repeat in range(5):
            converged = repeat != 4
            row = dict(case_id="mixed", n=100, d=4, k=2, seed=0, phase="measured",
                       method=method, external_fit_s=1.0, quality_valid=True,
                       steps=3 if converged else 30,
                       termination_reason="tolerance" if converged else "max_steps",
                       convergence_satisfied=converged,
                       final_centroid_displacement=0.0 if converged else 0.1,
                       quality_json=json.dumps({"relative_sse_gap": 0.0}))
            row.update({key: 0.1 for key in benchmark.TIMING_KEYS})
            rows.append(row)
        rows.append(dict(rows[-1], phase="warmup", steps=999, termination_reason="warmup-only"))
    summary = benchmark.summarize(rows)
    for method in ("cpu", "gpu"):
        assert json.loads(summary[f"{method}_iteration_counts"]) == [3, 3, 3, 3, 30]
        assert json.loads(summary[f"{method}_termination_reasons"]) == ["max_steps", "tolerance"]
        assert summary[f"{method}_converged_repeats"] == 4
        assert summary[f"{method}_convergence_satisfied"] is False
