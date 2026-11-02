"""Tests for the isolated exact-refinement experiment adapter."""

import numpy as np
import pytest
import csv
from pathlib import Path

from pcap.contracts import RunResult
from experiments.run_centroid_quality_refinement import (
    REFINEMENT_LEVELS,
    exact_refinement,
    refinement_config,
    _summarize,
    _write_final_report,
)


def test_refinement_config_is_limited_to_predeclared_levels():
    assert REFINEMENT_LEVELS == (0, 1, 3, 5)
    for level in (1, 3, 5):
        cfg = refinement_config(2, level)
        assert cfg.k == 16
        assert cfg.seed == 2
        assert cfg.max_steps == level
        assert cfg.tolerance == 0.0
        assert cfg.batch_size is None
        assert cfg.ann_effort is None


@pytest.mark.parametrize("level", [-1, 2, 4, 6])
def test_unapproved_refinement_levels_rejected(level):
    with pytest.raises(ValueError, match="refinement steps"):
        refinement_config(0, level)


def test_zero_refinement_is_a_true_noop_and_does_not_call_cuda():
    X = np.zeros((2, 4), dtype=np.float32)
    C = np.zeros((16, 4), dtype=np.float32)

    def forbidden(*args):
        raise AssertionError("zero-step refinement must not launch a fit")

    assert exact_refinement(X, C, 0, 0, fit_fn=forbidden) is None


def test_exact_refinement_passes_requested_cap_and_never_receives_target():
    X = np.zeros((3, 4), dtype=np.float32)
    C = np.zeros((16, 4), dtype=np.float32)
    seen = {}

    def fake_fit(points, centroids, config):
        seen["points"] = points
        seen["centroids"] = centroids
        seen["config"] = config
        return RunResult(centroids.copy(), np.zeros(len(points), dtype=np.int32), [],
                         {}, {"final_exact_sse": 0.0})

    result = exact_refinement(X, C, 1, 3, fit_fn=fake_fit)
    assert result.sse == 0.0
    assert seen["config"].max_steps == 3
    assert seen["config"].tolerance == 0.0
    assert not hasattr(seen["config"], "target_sse")
    assert seen["centroids"].flags.c_contiguous
    assert seen["centroids"].dtype == np.float32


def test_final_report_reads_nested_phase_two_records_and_keeps_target_counts(tmp_path: Path):
    phase2 = tmp_path / "phase2_exact_refinement"
    phase2.mkdir()
    with (phase2 / "individual_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method"])
        writer.writeheader()
        writer.writerow({"method": "G0_EXACT_CUDA"})
    fields = ["method", "refinement_requested", "median_end_to_end_wall_s", "median_final_exact_sse",
              "median_gap_to_g0_pct", "target_hits", "n_runs"]
    with (phase2 / "method_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "CUMULATIVE_FIXED_65536", "refinement_requested": 1,
                         "median_end_to_end_wall_s": 4, "median_final_exact_sse": 3,
                         "median_gap_to_g0_pct": 2, "target_hits": 1, "n_runs": 3})
        writer.writerow({"method": "G0_EXACT_CUDA", "refinement_requested": 0,
                         "median_end_to_end_wall_s": 1, "median_final_exact_sse": 2,
                         "median_gap_to_g0_pct": 0, "target_hits": 3, "n_runs": 3})
    _write_final_report(tmp_path)
    report = (tmp_path / "final_summary" / "final_summary.md").read_text()
    assert "1/3" in report
    assert "3/3" in report
    assert "G0_EXACT_CUDA" not in report


def test_summary_preserves_run_denominator(tmp_path: Path):
    runs = []
    for index, sse in enumerate((10.0, 11.0, 12.0)):
        runs.append({
            "method": "G0_EXACT_CUDA", "refinement_requested": "0",
            "end_to_end_wall_s": str(index + 1), "final_exact_sse": str(sse),
            "relative_gap_to_exact_cuda": "0", "frozen_target_hit": "True",
            "refinement_actual_steps": "2",
        })
    _summarize(tmp_path, runs)
    with (tmp_path / "method_summary.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["n"] == "3"
    assert row["n_runs"] == "3"
    assert row["target_hits"] == "3"
