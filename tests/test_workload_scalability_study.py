"""CPU-only protocol guards for the real-GPU workload-scaling runner."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import workload_scalability_study as study


def test_frozen_quality_rule_is_five_percent_of_exact_reference():
    assert study._target_sse(100.0) == 105.0
    assert study._target_sse(0.125) == pytest.approx(0.13125)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_quality_rule_rejects_invalid_reference(value):
    with pytest.raises(ValueError):
        study._target_sse(value)


def test_cumulative_and_decayed_fixed_widths_remain_separate():
    specs = study._method_specs(4_194_265)
    cumulative = [s for s in specs if s["family"] == "cumulative" and s["kind"] == "fixed"]
    decayed = [s for s in specs if s["family"] == "decayed" and s["kind"] == "fixed_decay"]
    assert [s["batch_size"] for s in cumulative] == list(study.BATCHES)
    assert [s["batch_size"] for s in decayed] == list(study.BATCHES)
    assert all(s["method"].startswith("CUM_FIXED_") for s in cumulative)
    assert all(s["method"].startswith("DECAY_FIXED_") for s in decayed)
    assert {s["method"] for s in specs} >= {"QGCUDA", "ADAPTIVE_V2"}


def test_largest_prepared_case_fits_conservative_vram_preflight():
    estimate = study._gpu_workspace_estimate(4_194_265, 4, 32)
    assert estimate > 4_194_265 * 4 * 4
    assert estimate < 200_000_000


def test_size_subset_arguments_are_supported_for_separate_bounded_processes():
    parser = study.argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", choices=tuple(study.SIZES), default=None)
    parser.add_argument("--k-values", type=int, nargs="+", choices=tuple(study.KS), default=None)
    args = parser.parse_args(["--sizes", "262144", "4194265"])
    assert args.sizes == [262144, 4194265]
    assert args.k_values is None


def test_k_subset_arguments_are_supported_for_focused_confirmation():
    parser = study.argparse.ArgumentParser()
    parser.add_argument("--k-values", type=int, nargs="+", choices=tuple(study.KS), default=None)
    args = parser.parse_args(["--k-values", "8", "16"])
    assert args.k_values == [8, 16]
