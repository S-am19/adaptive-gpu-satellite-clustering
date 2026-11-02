"""Experiment-only equal-observation Mini-Batch runs for controlled ablations.

The public production fit functions are unchanged. This runner reuses their
tested CPU/GPU state adapters and update kernels, but stops at an exact
processed-observation budget and scores a common disjoint fixed probe.
"""

from dataclasses import asdict, replace
import hashlib
from time import perf_counter

import numpy as np

from .adaptive_controller import BatchPolicy, Checkpoint, initial_state, transition
from .contracts import FitConfig, RunResult, validate_points_centroids
from .controlled_fit import _CPUState, _GPUState
from .quality_guard_controller import (
    QualityGuardPolicy, initial_quality_guard_state, quality_guard_transition,
)


def fit_budgeted_minibatch(X, C0, config, *, device="cuda", observation_budget,
                           adaptive=False, batch_policy=None, quality_guard_policy=None,
                           evaluation_interval=None, disjoint_probe=False,
                           diagnostic_trace=False):
    """Run exact-assignment cumulative updates to a precise observation budget.

    Evaluation target values are deliberately not accepted. Full-data SSE is
    returned as checkpoint data; target-hit calculations belong to the caller.
    The committed G1/G3 sampling stream is preserved by default. Set
    disjoint_probe=True for quality-guard experiments that require an
    independent monitor; that arm excludes the same fixed probe from sampling.
    """
    wall = perf_counter()
    validate_points_centroids(X, C0)
    if not isinstance(config, FitConfig) or config.batch_size is None:
        raise TypeError("config must be FitConfig with a batch_size")
    n, d = X.shape
    if C0.shape[0] != config.k or config.k > n:
        raise ValueError("C0/config.k mismatch")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if isinstance(observation_budget, bool) or not isinstance(observation_budget, (int, np.integer)) or observation_budget < 1:
        raise ValueError("observation_budget must be positive")
    budget = int(observation_budget)
    if evaluation_interval is None:
        evaluation_interval = budget
    if isinstance(evaluation_interval, bool) or not isinstance(evaluation_interval, (int, np.integer)) or evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive")
    if adaptive and quality_guard_policy is not None:
        raise ValueError("choose the committed batch policy or the new quality guard, not both")
    if quality_guard_policy is not None and not disjoint_probe:
        raise ValueError("quality-guarded runs require a disjoint fixed probe")
    if adaptive:
        batch_policy = batch_policy or BatchPolicy.for_run(n, config.batch_size)
        if batch_policy.initial_batch != config.batch_size or batch_policy.maximum > n:
            raise ValueError("batch policy must match initial batch and fit the dataset")
    elif batch_policy is not None:
        raise ValueError("batch_policy requires adaptive=True")
    if quality_guard_policy is not None:
        if not isinstance(quality_guard_policy, QualityGuardPolicy):
            raise TypeError("quality_guard_policy must be QualityGuardPolicy")
        if quality_guard_policy.initial_batch != config.batch_size or quality_guard_policy.maximum_batch > n:
            raise ValueError("quality-guard bounds must match initial batch and dataset")
    max_capacity = (quality_guard_policy.maximum_batch if quality_guard_policy else
                    batch_policy.maximum if adaptive else config.batch_size)
    if max_capacity * int(config.max_steps) > np.iinfo(np.int64).max:
        raise ValueError("observation capacity exceeds int64")
    if budget > int(config.max_steps) * max_capacity:
        raise ValueError("budget exceeds max_steps times maximum batch")

    # Separate deterministic stream; held-out indices never enter a batch.
    probe_rng = np.random.default_rng(np.random.SeedSequence([int(config.seed), 1]))
    probe_size = min(n, 1024)
    if probe_size == n:
        probe_size = max(1, n // 20)
    probe_indices = np.sort(probe_rng.choice(n, size=probe_size, replace=False))
    if disjoint_probe:
        training_mask = np.ones(n, dtype=np.bool_)
        training_mask[probe_indices] = False
        training_indices = np.flatnonzero(training_mask).astype(np.int32)
    else:
        training_indices = np.arange(n, dtype=np.int32)
    if max_capacity > len(training_indices):
        raise ValueError("batch capacity exceeds the disjoint training population")
    probe = np.ascontiguousarray(X[probe_indices])
    rng = np.random.default_rng(int(config.seed))

    setup_start = perf_counter()
    if device == "cuda":
        from numba import cuda
        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError("real CUDA is required; simulator/fallback is disabled")
        state = _GPUState(X, C0, max_capacity, int(config.seed), False)
        state.device.set_probe(probe_indices)
    else:
        state = _CPUState(X, C0, max_capacity, int(config.seed), False)
    setup_s = perf_counter() - setup_start
    checkpoint_centroids = C0.copy()
    previous_labels, initial_sse, _ = state.probe(probe_indices, checkpoint_centroids)
    scale = max(float(np.sqrt(np.mean(np.sum((probe.astype(np.float64)-probe.mean(axis=0))**2, axis=1)))), 1e-12)
    monitor_s = controller_s = sampling_s = assignment_s = update_s = quality_s = 0.0
    batch_state = None
    q_state = None
    if adaptive:
        batch_state = initial_state(batch_policy, initial_sse / probe_size)
    if quality_guard_policy:
        q_state = initial_quality_guard_state(quality_guard_policy, initial_sse / probe_size)
    history, checkpoints, events = [], [], []
    diagnostic_previous_C = C0.copy() if diagnostic_trace else None
    observations = 0
    rolled_back_observations = 0
    step = 0
    next_eval = min(int(evaluation_interval), budget)
    first_eval = True
    target_unaware = quality_guard_policy is not None
    # Safe model and sufficient-statistic snapshot for transactional rollback.
    if target_unaware:
        if device == "cuda":
            from numba import cuda
            safe_C = cuda.device_array_like(state.device.C)
            safe_sums = cuda.device_array_like(state.device.sums)
            safe_counts = cuda.device_array_like(state.device.counts)
            safe_C.copy_to_device(state.device.C); safe_sums.copy_to_device(state.device.sums)
            safe_counts.copy_to_device(state.device.counts); cuda.synchronize()
        else:
            safe_C, safe_sums, safe_counts = state.C.copy(), state.sums.copy(), state.counts.copy()
        safe_probe_sse = initial_sse
        safe_observation_count = 0
        rolled_back_observations = 0

    while observations < budget and step < int(config.max_steps):
        step += 1
        requested = (int(batch_state.batch_size) if batch_state else
                     int(q_state.batch_size) if q_state else int(config.batch_size))
        to_eval = next_eval - observations if observations < next_eval else requested
        actual = min(requested, budget - observations, to_eval)
        t = perf_counter()
        draw = rng.choice(len(training_indices), size=actual, replace=False)
        indices = training_indices[draw]
        sampling_s += perf_counter() - t
        sample_hash = (hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int32)
                                      .view(np.uint8)).hexdigest()
                       if diagnostic_trace else None)
        t = perf_counter(); batch_sse = state.assign(indices, None); assignment_s += perf_counter()-t
        t = perf_counter(); movement = state.update(False); update_s += perf_counter()-t
        observations += actual
        entry = dict(step=step, phase="adapting" if adaptive or target_unaware else "fixed",
                     requested_batch_size=requested, actual_batch_size=actual,
                     points_processed=observations, batch_sse_before_update=batch_sse,
                     centroid_displacement=movement, probe_checked=False,
                     controller_decision=None, rollback=False)
        if diagnostic_trace:
            entry["sample_indices_sha256"] = sample_hash
        should_probe = step % int(config.probe_interval) == 0 or observations == next_eval or observations == budget
        if should_probe:
            t = perf_counter()
            labels, probe_sse, probe_movement = state.probe(probe_indices, checkpoint_centroids)
            changes = float(np.mean(labels != previous_labels))
            previous_labels = labels
            if device == "cpu":
                checkpoint_centroids = state.C.copy()
            monitor_s += perf_counter() - t
            entry.update(probe_checked=True, probe_sse=probe_sse,
                         normalized_probe_movement=probe_movement / scale,
                         probe_label_change_fraction=changes)
            if target_unaware:
                t = perf_counter()
                q_state, decision = quality_guard_transition(
                    quality_guard_policy, q_state, probe_sse / probe_size,
                    probe_movement / scale, changes)
                controller_s += perf_counter() - t
                if decision.rollback:
                    t = perf_counter()
                    if device == "cuda":
                        before_count = int(np.sum(state.device.counts.copy_to_host(), dtype=np.int64))
                        state.device.C.copy_to_device(safe_C)
                        state.device.sums.copy_to_device(safe_sums)
                        state.device.counts.copy_to_device(safe_counts)
                        from numba import cuda
                        cuda.synchronize()
                        labels, restored_sse, _ = state.probe(probe_indices, None)
                    else:
                        before_count = int(np.sum(state.counts, dtype=np.int64))
                        state.C[...] = safe_C; state.sums[...] = safe_sums; state.counts[...] = safe_counts
                        labels, restored_sse, _ = state.probe(probe_indices, checkpoint_centroids)
                    discarded = max(0, before_count - safe_observation_count)
                    rolled_back_observations += discarded
                    previous_labels = labels
                    if device == "cpu":
                        checkpoint_centroids = state.C.copy()
                    monitor_s += perf_counter() - t
                    entry["rollback"] = True
                    entry["restored_probe_sse"] = restored_sse
                    q_state = replace(q_state, previous_objective=safe_probe_sse / probe_size)
                    events.append(dict(step=step, points_processed=observations,
                                       event="rollback", reason=decision.reason,
                                       from_probe_sse=probe_sse, restored_probe_sse=restored_sse,
                                       observations_discarded=discarded,
                                       restored_cumulative_observations=safe_observation_count,
                                       batch_size_before=decision.previous_batch,
                                       batch_size_after=decision.next_batch))
                else:
                    # Only a non-deteriorating state can become the rollback point.
                    if probe_sse <= safe_probe_sse:
                        safe_probe_sse = probe_sse
                        if device == "cuda":
                            safe_C.copy_to_device(state.device.C); safe_sums.copy_to_device(state.device.sums)
                            safe_counts.copy_to_device(state.device.counts)
                            from numba import cuda
                            cuda.synchronize()
                        else:
                            safe_C, safe_sums, safe_counts = state.C.copy(), state.sums.copy(), state.counts.copy()
                        safe_observation_count = (int(np.sum(state.device.counts.copy_to_host(), dtype=np.int64))
                                                  if device == "cuda" else int(np.sum(state.counts, dtype=np.int64)))
                entry["controller_decision"] = asdict(decision)
            elif adaptive and step % int(config.probe_interval) == 0:
                t = perf_counter()
                checkpoint = Checkpoint(probe_sse / probe_size, probe_movement / scale, changes)
                batch_state, decision = transition(batch_policy, batch_state, checkpoint)
                controller_s += perf_counter() - t
                entry["controller_decision"] = asdict(decision)
                entry["next_batch_size"] = int(batch_state.batch_size)
            history.append(entry)
        else:
            history.append(entry)

        if observations == next_eval or observations == budget:
            t = perf_counter(); exact_sse = state.objective(); quality_s += perf_counter()-t
            checkpoint = dict(points_processed=observations, full_exact_sse=exact_sse,
                              step=step, elapsed_s=perf_counter()-wall,
                              checkpoint_kind="budget" if observations == budget else "observation_interval")
            if diagnostic_trace:
                trace_start = perf_counter()
                if device == "cuda":
                    from numba import cuda
                    current_C = state.device.C.copy_to_host()
                    current_counts = state.device.counts.copy_to_host()
                    cuda.synchronize()
                else:
                    current_C = state.C.copy()
                    current_counts = state.counts.copy()
                delta = current_C.astype(np.float64) - diagnostic_previous_C.astype(np.float64)
                checkpoint.update(
                    cluster_update_counts=np.asarray(current_counts, dtype=np.int64).tolist(),
                    checkpoint_centroid_movement=float(np.sqrt(np.max(np.sum(delta * delta, axis=1)))),
                    per_cluster_centroid_movement=np.sqrt(np.sum(delta * delta, axis=1)).tolist(),
                )
                diagnostic_previous_C = current_C.copy()
                monitor_s += perf_counter() - trace_start
            checkpoints.append(checkpoint)
            next_eval = min(next_eval + int(evaluation_interval), budget)

    if observations < budget:
        termination = "max_steps"
    else:
        termination = "observation_budget"
    t = perf_counter(); C, labels, counts, final_sse = state.final(); final_s = perf_counter()-t
    total = perf_counter() - wall
    if checkpoints and checkpoints[-1]["points_processed"] == observations:
        checkpoints[-1]["full_exact_sse"] = final_sse
    timings = dict(wall_s=total, setup_s=setup_s, sampling_s=sampling_s,
                   batch_assignment_s=assignment_s, accumulation_update_s=update_s,
                   monitoring_s=monitor_s, controller_s=controller_s,
                   quality_evaluation_s=quality_s, final_assignment_s=final_s,
                   compute_s=total-setup_s)
    metadata = dict(algorithm="budgeted_" + ("quality_guarded" if target_unaware else "adaptive" if adaptive else "fixed") + "_" + device,
        execution_device=device, config=asdict(config), policy=asdict(quality_guard_policy or batch_policy) if (quality_guard_policy or batch_policy) else None,
        observation_budget=budget, points_processed=observations, steps=step,
        termination_reason=termination, convergence_satisfied=False,
        final_exact_sse=final_sse, final_exact_evaluation=True,
        cluster_observation_counts=np.asarray(counts).tolist(),
        probe_indices=probe_indices.tolist(), probe_size=probe_size, probe_is_disjoint=bool(disjoint_probe),
        committed_observations=int(np.sum(counts, dtype=np.int64)),
        observations_rolled_back=rolled_back_observations,
        diagnostic_trace_enabled=bool(diagnostic_trace),
        training_population=len(training_indices), sampling_rng="numpy.default_rng(seed)",
        probe_rng="numpy.default_rng(SeedSequence([seed,1]))",
        sampling_policy=("uniform without replacement per batch over complement of fixed probe; independent draws across updates"
                         if disjoint_probe else "uniform without replacement per batch over all X; independent draws across updates"),
        update_rule="cumulative float64 sums/int64 counts; empty clusters retain prior centroid",
        quality_guard_uses_evaluation_target=False,
        state_rollback="centroids, cumulative sums and counts restored from best non-deteriorating fixed-probe checkpoint" if target_unaware else None,
        target_sse_input=False)
    if device == "cuda":
        from numba import cuda
        cuda.synchronize(); metadata["gpu"] = str(cuda.get_current_device().name)
        metadata["estimated_device_buffer_bytes"] = state.device.estimated_bytes
    return RunResult(C, labels, history, timings, metadata), checkpoints, events
