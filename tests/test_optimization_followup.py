"""Tests for derived, non-fit analysis and archive verification helpers."""
from pathlib import Path

import pytest

from experiments.finalize_adaptive_optimization_followup import (
    median_iqr,
    summarize_timing,
    verify_checksums,
)


def test_median_iqr_matches_numpy_percentile_convention():
    assert median_iqr([1, 2, 3, 4, 5]) == (3.0, 2.0)


def test_timing_summary_keeps_wall_and_internal_categories_separate():
    rows = [
        {"method": "A", "wall": "10", "assignment": "4", "update": "5"},
        {"method": "A", "wall": "12", "assignment": "6", "update": "5"},
    ]
    result = summarize_timing(
        rows, {"A"}, "wall", {"assignment": "assignment", "update": "update"}
    )[0]
    assert result["external_wall_median_s"] == 11.0
    assert result["assignment_median_s"] == 5.0
    assert result["update_median_s"] == 5.0
    # The helper reports fields independently; it does not add overlapping
    # categories to synthesize an end-to-end measurement.
    assert "internal_total_s" not in result


def test_checksum_verifier_accepts_valid_and_rejects_changed_artifact(tmp_path: Path):
    import hashlib

    artifact = tmp_path / "evidence.csv"
    artifact.write_bytes(b"measured,data\n1,2\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{digest}  evidence.csv\n")
    assert verify_checksums(tmp_path) == 1
    artifact.write_bytes(b"modified\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path)
