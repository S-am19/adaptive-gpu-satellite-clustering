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


def test_interleaved_schedule_balances_sizes_methods_and_warms_first():
    args = benchmark.parse_args(["--sizes", "8192", "32768", "--dimensions", "14",
                                 "--clusters", "16", "--seeds", "0", "--warmups", "2",
                                 "--interleave-sizes"])
    schedule = list(benchmark.case_schedule(args))
    assert len(schedule) == 14
    assert all(phase == "warmup" for _, phase, _, _ in schedule[:4])
    assert all(phase == "measured" for _, phase, _, _ in schedule[4:])
    for repeat in range(5):
        round_ = schedule[4 + 2 * repeat:6 + 2 * repeat]
        assert [case[0] for case, _, _, _ in round_] == (
            [8192, 32768] if repeat % 2 == 0 else [32768, 8192])
        assert round_[0][3] != round_[1][3]
    for n in (8192, 32768):
        rows = [row for row in schedule if row[0][0] == n]
        assert sum(row[1] == "warmup" for row in rows) == 2
        measured = [row for row in rows if row[1] == "measured"]
        assert sorted(row[2] for row in measured) == list(range(5))
        assert sum(row[3][0] == "cpu" for row in measured) in (2, 3)


def test_default_schedule_preserves_case_major_order():
    args = benchmark.parse_args(["--sizes", "8192", "32768", "--dimensions", "14",
                                 "--clusters", "16", "--seeds", "0"])
    schedule = list(benchmark.case_schedule(args))
    assert [row[0][0] for row in schedule] == [8192] * 6 + [32768] * 6


@pytest.fixture
def mocked_runner(monkeypatch, tmp_path):
    """Exercise runner I/O and control flow without CUDA or measured algorithms."""
    import sys
    import types
    import weakref
    from numba import cuda
    import pcap.kmeans_cpu as cpu_module

    monkeypatch.setattr(benchmark, '_configure_process', lambda: None)
    monkeypatch.setattr(benchmark, '_command', lambda args: {'returncode': 0, 'stdout': ''})
    monkeypatch.setattr(benchmark, '_available_ram', lambda: 10**8)
    monkeypatch.setattr(benchmark.gc, 'collect', lambda: None)
    monkeypatch.setattr(cuda.config, 'ENABLE_CUDASIM', False)
    monkeypatch.setattr(cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(cuda, 'current_context', lambda: types.SimpleNamespace(
        get_memory_info=lambda: (10**8, 10**8)))
    clock = [0.0]
    monkeypatch.setattr(benchmark, 'perf_counter', lambda: clock[0])
    references, live_at_load = [], []

    def generate(n, d, k, seed):
        live_at_load.append(sum(ref() is not None for ref in references))
        points = np.zeros((n, d), dtype=np.float32)
        references.append(weakref.ref(points))
        return points

    calls = []
    fail_at = [None]

    def fit(points, initial, config):
        calls.append(len(points))
        clock[0] += 0.001
        if len(calls) == fail_at[0]:
            raise RuntimeError('injected fit failure')
        return RunResult(initial.copy(), np.zeros(len(points), dtype=np.int32),
                         [{'centroid_displacement': 0.0}],
                         {key: 0.001 for key in benchmark.TIMING_KEYS},
                         {'steps': 1, 'termination_reason': 'tolerance', 'final_exact_sse': 0.0})

    monkeypatch.setattr(benchmark, 'make_synthetic', generate)
    monkeypatch.setattr(cpu_module, 'fit_cpu', fit)
    monkeypatch.setitem(sys.modules, 'pcap.kmeans_cuda', types.SimpleNamespace(fit_cuda=fit))
    args = ['--sizes', '8', '16', '--dimensions', '2', '--clusters', '1', '--seeds', '0',
            '--output-dir', str(tmp_path / 'artifacts')]
    return types.SimpleNamespace(args=args, output=tmp_path / 'artifacts', clock=clock,
                                 references=references, live_at_load=live_at_load,
                                 calls=calls, fail_at=fail_at)


def read_runner_artifacts(run):
    import csv
    manifest = json.loads((run.output / 'manifest.json').read_text())
    raw = list(csv.DictReader((run.output / 'raw.csv').open()))
    path = run.output / 'summary.csv'
    summaries = list(csv.DictReader(path.open())) if path.exists() else []
    return manifest, raw, summaries


@pytest.mark.parametrize('interleaved', [False, True])
@pytest.mark.parametrize('failure', ['timeout', 'exception'])
def test_completed_summary_survives_interruption(monkeypatch, mocked_runner, interleaved, failure):
    run = mocked_runner
    original = benchmark.case_schedule

    def interrupted(args):
        completed = False
        for entry in original(args):
            if completed:
                if failure == 'exception':
                    raise RuntimeError('injected schedule failure')
                run.clock[0] = 121.0
            yield entry
            _, phase, repeat, _ = entry
            if phase == 'measured' and repeat == args.repeats - 1:
                completed = True

    # Manifest schedule enumeration must not trigger the injected execution failure.
    invocations = [0]
    def schedule(args):
        invocations[0] += 1
        return original(args) if invocations[0] == 1 else interrupted(args)
    monkeypatch.setattr(benchmark, 'case_schedule', schedule)
    with pytest.raises(TimeoutError if failure == 'timeout' else RuntimeError):
        benchmark.main(run.args + (['--interleave-sizes'] if interleaved else []))
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['status'] == 'failed'
    assert manifest['completed_cases'] == len(summaries) == 1
    completed_id = summaries[0]['case_id']
    assert len([row for row in raw if row['case_id'] == completed_id]) == 12
    if interleaved:
        incomplete = [row for row in raw if row['case_id'] != completed_id]
        assert len(incomplete) == 10
        assert sum(row['phase'] == 'measured' for row in incomplete) == 8


@pytest.mark.parametrize('interleaved', [False, True])
def test_mid_pair_failure_preserves_partial_raw_without_completion(mocked_runner, interleaved):
    run = mocked_runner
    run.fail_at[0] = 4  # One full pair, then one successful call before failure.
    with pytest.raises(RuntimeError, match='injected fit failure'):
        benchmark.main(run.args + (['--interleave-sizes'] if interleaved else []))
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['completed_cases'] == 0 and summaries == []
    assert len(raw) == 3
    assert raw[-1]['quality_valid'] == 'False'
    assert json.loads(raw[-1]['quality_json']) == {'pair_incomplete': True}


@pytest.mark.parametrize('interleaved,expected_live', [(False, [0, 0]), (True, [0, 1])])
def test_dataset_lifetime_and_completed_artifacts(mocked_runner, interleaved, expected_live):
    run = mocked_runner
    benchmark.main(run.args + (['--interleave-sizes'] if interleaved else []))
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['status'] == 'complete' and manifest['completed_cases'] == 2
    assert len(summaries) == 2 and len(raw) == 24
    assert manifest['final_cuda_memory_info'] == [10**8, 10**8]
    assert 'final_cuda_memory_error' not in manifest
    assert run.live_at_load == expected_live
    assert all(ref() is None for ref in run.references)
    assert all(row['repeats'] == '5' for row in summaries)


def test_interleaved_memory_guard_counts_retained_cases(monkeypatch, mocked_runner):
    run = mocked_runner
    # Each case fits individually under 2500 bytes; their summed estimates do not.
    monkeypatch.setattr(benchmark, '_available_ram', lambda: 10000)
    with pytest.raises(MemoryError, match='RAM budget'):
        benchmark.main(run.args + ['--interleave-sizes'])
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['completed_cases'] == 0 and summaries == []
    assert len(run.references) == 1 and len(raw) == 2


@pytest.mark.parametrize('interleaved', [False, True])
def test_cleanup_cuda_failure_preserves_original_error_and_artifacts(monkeypatch, mocked_runner, interleaved):
    from numba import cuda
    run = mocked_runner
    # Fail the second call of a pair, after one case has fully completed.
    run.fail_at[0] = 24 if interleaved else 14
    original_context = cuda.current_context

    def context():
        if len(run.calls) >= run.fail_at[0]:
            raise RuntimeError('CUDA context unavailable during cleanup')
        return original_context()

    monkeypatch.setattr(cuda, 'current_context', context)
    with pytest.raises(RuntimeError, match='^injected fit failure$'):
        benchmark.main(run.args + (['--interleave-sizes'] if interleaved else []))
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['status'] == 'failed'
    assert manifest['error'] == 'RuntimeError: injected fit failure'
    assert manifest['completed_cases'] == len(summaries) == 1
    assert manifest['final_cuda_memory_info'] is None
    assert manifest['final_cuda_memory_error'] == 'RuntimeError: CUDA context unavailable during cleanup'
    assert len(raw) == run.fail_at[0] - 1
    assert raw[-1]['quality_valid'] == 'False'
    assert json.loads(raw[-1]['quality_json']) == {'pair_incomplete': True}
    assert len([row for row in raw if row['case_id'] == summaries[0]['case_id']]) == 12


def test_successful_run_survives_unavailable_final_memory_reading(monkeypatch, mocked_runner):
    from numba import cuda
    run = mocked_runner
    original_context = cuda.current_context

    def context():
        if len(run.calls) == 24:
            raise RuntimeError('final memory reading unavailable')
        return original_context()

    monkeypatch.setattr(cuda, 'current_context', context)
    assert benchmark.main(run.args) == run.output
    manifest, raw, summaries = read_runner_artifacts(run)
    assert manifest['status'] == 'complete' and 'error' not in manifest
    assert manifest['completed_cases'] == len(summaries) == 2
    assert len(raw) == 24
    assert manifest['final_cuda_memory_info'] is None
    assert manifest['final_cuda_memory_error'] == 'RuntimeError: final memory reading unavailable'
