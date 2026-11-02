from experiments.run_adaptive_four_phase import _first_target_crossing


def test_exact_cuda_target_crossing_returns_observations_not_seconds():
    history = [
        {"step": 0, "points_processed": 0, "full_data_sse": 100.0, "elapsed_s": 0.01},
        {"step": 1, "points_processed": 1000, "full_data_sse": 70.0, "elapsed_s": 0.02},
        {"step": 2, "points_processed": 2000, "full_data_sse": 50.0, "elapsed_s": 0.03},
    ]
    assert _first_target_crossing("G0", history, [], 60.0, 1000) == (2000, 0.03)


def test_minibatch_target_crossing_uses_first_exact_checkpoint():
    checkpoints = [
        {"points_processed": 8192, "full_exact_sse": 90.0, "elapsed_s": 0.4},
        {"points_processed": 16384, "full_exact_sse": 50.0, "elapsed_s": 0.8},
    ]
    assert _first_target_crossing("G1", [], checkpoints, 60.0, 1000) == (16384, 0.8)


def test_unreached_target_has_no_crossing():
    assert _first_target_crossing("G0", [{"full_data_sse": 101.0, "elapsed_s": 1.0}], [], 100.0, 1000) == (None, None)
