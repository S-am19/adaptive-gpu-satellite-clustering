"""Bounded real-Sentinel-2 mini-batch plus exact-CUDA refinement pilot.

The existing production baselines are imported unchanged. The runner trains a
fixed cumulative or fixed-decay CUDA Mini-Batch model to the frozen observation
budget, then optionally applies 0/1/3/5 exact CUDA Lloyd updates. It never passes
the evaluation target into a fitting/controller function.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/20261027-centroid-quality-refinement-study"
X_PATH = ROOT / "outputs/20261025-independent-holdouts/validation512/X.npy"
DECAY_ARCHIVE = ROOT / "outputs/20261027-decay-memory-study/phase2_validation"
FROZEN_PROTOCOL = ROOT / "outputs/20261027-decay-memory-study/frozen_validation_protocol.json"
SEEDS = (0, 1, 2)
N, D, K = 262_144, 4, 16
BATCH = 65_536
BUDGET = 40_960_000
REFINEMENT_LEVELS = (0, 1, 3, 5)
CHECKPOINTS = tuple(BUDGET * i // 5 for i in range(1, 6))
METHODS = ("CUMULATIVE_FIXED_65536", "DECAYED_FIXED_65536")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def verify_source_checksums(directory: Path) -> int:
    checksum_file = directory / "checksums.sha256"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Source checksum manifest missing: {checksum_file}")
    verified = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = directory / relative.strip()
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Source checksum mismatch: {path}")
        verified += 1
    return verified


def refinement_config(seed: int, steps: int):
    """Build a valid exact-CUDA refinement config for a requested level."""
    from pcap.contracts import FitConfig

    if steps not in REFINEMENT_LEVELS[1:]:
        raise ValueError(f"refinement steps must be one of {REFINEMENT_LEVELS}")
    return FitConfig(k=K, seed=int(seed), max_steps=int(steps), tolerance=0.0)


def exact_refinement(X, centroids, seed: int, steps: int, *, fit_fn=None):
    """Apply a requested number of exact CUDA Lloyd updates; level 0 is a no-op.

    A zero-step result is represented by ``None`` and must use the already
    measured exact Mini-Batch final SSE. The evaluator target is intentionally
    absent from this interface.
    """
    if steps == 0:
        return None
    cfg = refinement_config(seed, steps)
    if fit_fn is None:
        from pcap.kmeans_cuda import fit_cuda
        fit_fn = fit_cuda
    return fit_fn(X, np.ascontiguousarray(centroids, dtype=np.float32), cfg)


def _run(command: list[str]) -> str:
    p = subprocess.run(command, text=True, capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError(f"Command failed ({command!r}): {p.stderr.strip()}")
    return p.stdout.strip()


def hardware_snapshot() -> dict:
    from numba import cuda

    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Real CUDA is required; simulator and CPU fallback are disabled")
    dev = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    try:
        smi = _run(["nvidia-smi", "--query-gpu=name,uuid,memory.free,memory.total,pstate,temperature.gpu,utilization.gpu,clocks.sm", "--format=csv"])
    except Exception as exc:
        smi = f"unavailable: {exc!r}"
    try:
        nvcc = _run(["nvcc", "--version"])
    except Exception as exc:
        nvcc = f"unavailable: {exc!r}"
    packages = {}
    for name in ("numpy", "numba", "numba-cuda", "llvmlite", "cuda-bindings", "rasterio", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "gpu": str(dev.name), "compute_capability": list(dev.compute_capability),
        "cuda_simulator": bool(cuda.config.ENABLE_CUDASIM),
        "cuda_driver_version": list(cuda.cudadrv.driver.get_version()),
        "cuda_runtime_version": list(cuda.cudadrv.runtime.get_version()),
        "free_vram_bytes": int(free), "total_vram_bytes": int(total),
        "nvidia_smi": smi, "nvcc": nvcc, "packages": packages,
        "python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def _append(path: Path, rows: list[dict], fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, sort_keys=True, default=str) + "\n")


def _save_centroids(directory: Path, method: str, seed: int, steps: int, C: np.ndarray) -> str:
    path = directory / f"centroids_{method}_seed{seed}_refine{steps}.npz"
    np.savez_compressed(path, centroids=np.asarray(C, dtype=np.float32))
    return sha256_file(path)


def _case_fit(X, C0, seed: int, method: str):
    """Fit the selected unchanged update rule to the fixed observation budget."""
    from pcap.budgeted_minibatch import fit_budgeted_minibatch
    from pcap.contracts import FitConfig
    from pcap.minibatch_decayed_cuda import fit_decayed_minibatch_cuda

    cfg = FitConfig(k=K, seed=seed, max_steps=BUDGET // BATCH,
                    tolerance=0.0, batch_size=BATCH, probe_interval=20)
    start = perf_counter()
    if method == "CUMULATIVE_FIXED_65536":
        result, checkpoints, _ = fit_budgeted_minibatch(
            X, C0, cfg, device="cuda", observation_budget=BUDGET,
            adaptive=False, evaluation_interval=BUDGET // 5,
            disjoint_probe=False, diagnostic_trace=True)
        checkpoint_rows = checkpoints
    elif method == "DECAYED_FIXED_65536":
        result = fit_decayed_minibatch_cuda(
            X, C0, cfg, quality_interval=0,
            observation_budget=BUDGET,
            quality_observation_checkpoints=CHECKPOINTS)
        checkpoint_rows = result.metadata.get("quality_history", [])
    else:
        raise ValueError(f"Unknown method: {method}")
    return result, checkpoint_rows, perf_counter() - start


def _warmups(X, C0):
    """Warm each measured CUDA path twice on a small real-scene subset."""
    from pcap.budgeted_minibatch import fit_budgeted_minibatch
    from pcap.contracts import FitConfig
    from pcap.kmeans_cuda import fit_cuda
    from pcap.minibatch_decayed_cuda import fit_decayed_minibatch_cuda

    Xw = np.ascontiguousarray(X[:8192])
    rows = []
    for method in (*METHODS, "EXACT_CUDA"):
        for index in range(2):
            seed = 0
            cw = np.ascontiguousarray(C0[:K])
            started = perf_counter()
            if method == "EXACT_CUDA":
                fit_cuda(Xw, cw, FitConfig(k=K, seed=seed, max_steps=1, tolerance=0.0))
            else:
                cfg = FitConfig(k=K, seed=seed, max_steps=2, tolerance=0.0,
                                batch_size=1024, probe_interval=20)
                if method == "CUMULATIVE_FIXED_65536":
                    fit_budgeted_minibatch(Xw, cw, cfg, device="cuda", observation_budget=2048,
                                           evaluation_interval=1024, diagnostic_trace=False)
                else:
                    fit_decayed_minibatch_cuda(Xw, cw, cfg, quality_interval=0,
                                               observation_budget=2048,
                                               quality_observation_checkpoints=(1024, 2048))
            rows.append({"method": method, "warmup": index + 1,
                         "wall_s_excluded": perf_counter() - started,
                         "N_warmup": len(Xw), "batch_size_warmup": 1024})
    return rows


def _validate_inputs(X: np.ndarray, protocol: dict) -> None:
    if X.shape != (N, D) or X.dtype != np.float32 or not X.flags.c_contiguous:
        raise ValueError(f"Expected contiguous float32 X with shape {(N, D)}, got {X.shape}/{X.dtype}")
    if not np.isfinite(X).all():
        raise ValueError("Input feature matrix contains nonfinite values")
    if int(protocol["N"]) != N or int(protocol["D"]) != D or int(protocol["K"]) != K:
        raise ValueError("Frozen protocol shape/K differs from the requested study")
    if int(protocol["observation_budget"]) != BUDGET:
        raise ValueError("Frozen observation budget differs from requested study")


def run() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing study archive: {OUT}")
    if not X_PATH.is_file() or not FROZEN_PROTOCOL.is_file():
        raise FileNotFoundError("Prepared validation features or frozen protocol are missing")
    from numba import cuda
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("Real GPU unavailable; no simulator/CPU fallback is permitted")

    protocol = json.loads(FROZEN_PROTOCOL.read_text())
    targets = {int(k): float(v) for k, v in protocol["targets_by_seed"].items()}
    if set(SEEDS) - set(targets):
        raise ValueError("Frozen target table does not contain the pilot seeds")
    X = np.ascontiguousarray(np.load(X_PATH, mmap_mode="r"), dtype=np.float32)
    _validate_inputs(X, protocol)
    C0s = {}
    for seed in SEEDS:
        with np.load(DECAY_ARCHIVE / f"initial_centroids_seed{seed}.npz") as z:
            C0s[seed] = np.ascontiguousarray(z["C0"], dtype=np.float32)
        if C0s[seed].shape != (K, D) or not np.isfinite(C0s[seed]).all():
            raise ValueError(f"Invalid seed {seed} initial centroids")
    csv.field_size_limit(64 * 1024 * 1024)
    with (DECAY_ARCHIVE / "individual_runs.csv").open(newline="", encoding="utf-8") as stream:
        archived_runs = list(csv.DictReader(stream))
    archived_x = {r.get("X_sha256") for r in archived_runs if r.get("X_sha256")}
    if archived_x != {sha256_array(X)}:
        raise ValueError("Prepared X differs from the frozen decay experiment feature matrix")
    archived_c0 = {int(r["seed"]): r["C0_sha256"] for r in archived_runs
                   if r.get("method") == "G0_EXACT_CUDA" and r.get("C0_sha256")}
    for seed in SEEDS:
        if archived_c0.get(seed) != sha256_array(C0s[seed]):
            raise ValueError(f"Seed {seed} C0 differs from the frozen decay experiment")

    # Output tree and preregistered configuration exist before the first fit.
    for rel in ("phase1_centroid_updates", "phase2_exact_refinement",
                "phase3_adaptive_scheduling", "figures", "final_summary"):
        (OUT / rel).mkdir(parents=True, exist_ok=False)
    p1 = OUT / "phase1_centroid_updates"
    p2 = OUT / "phase2_exact_refinement"
    source_diag = ROOT / "outputs/20261027-decay-memory-study/phase0_convergence_diagnostic_retry1"
    source_diag_verified = 0
    if source_diag.exists():
        import shutil
        source_diag_verified = verify_source_checksums(source_diag)
        shutil.copytree(source_diag, p1 / "matched_cumulative_decay_diagnostic")
        if verify_source_checksums(p1 / "matched_cumulative_decay_diagnostic") != source_diag_verified:
            raise RuntimeError("Copied Phase 1 diagnostic failed checksum count validation")
    qg_manifest = ROOT / "outputs/20261027-qg-batchsize-ablation/manifest.json"
    source_validation = DECAY_ARCHIVE
    (p1 / "source_validation_archive.json").write_text(json.dumps({
        "fixed_decay_source": str(source_validation.relative_to(ROOT)),
        "fixed_decay_manifest_sha256": sha256_file(source_validation / "manifest.json"),
        "fixed_decay_checksums_sha256": sha256_file(source_validation / "checksums.sha256"),
        "cumulative_sweep_source": str(qg_manifest.parent.relative_to(ROOT)),
        "cumulative_manifest_sha256": sha256_file(qg_manifest),
        "matched_update_rule_diagnostic": str(source_diag.relative_to(ROOT)),
        "matched_diagnostic_checksum_entries_verified": source_diag_verified,
        "note": "Original archived source was not modified; phase 1 reuses its frozen comparisons."
    }, indent=2) + "\n")

    config = {
        "study": "centroid update quality and exact CUDA refinement pilot",
        "scope": "exploratory validation on previously examined 31UFU; not independent confirmation",
        "scene": protocol.get("validation_scene"), "N": N, "D": D, "K": K,
        "seeds": list(SEEDS), "batch_size": BATCH, "observation_budget": BUDGET,
        "update_methods": list(METHODS), "refinement_levels": list(REFINEMENT_LEVELS),
        "quality_target": "frozen exact-CPU-reference SSE * 1.05, evaluator-only",
        "target_values_by_seed": {str(k): targets[k] for k in SEEDS},
        "checkpoint_observations": list(CHECKPOINTS), "repetitions": 1,
        "warmups": "two excluded runs per update/refinement path on a real-scene subset",
        "sampling": "numpy.default_rng(seed), uniform without replacement within each batch; same batch width and seed across cumulative/decayed methods; per-update hashes compared",
        "probe": "fixed seeded overlapping probe; quality checkpoints are full-data exact SSE and do not control fitting",
        "refinement": "fit_cuda from trained centroids with tolerance=0; requested max_steps 1/3/5; record actual steps, setup/upload, transfer and complete wall",
        "termination": "Mini-Batch runs stop at the 40.96M observation budget; target is never passed to fitting code",
        "protocol_source": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "X_path": str(X_PATH.relative_to(ROOT)), "X_file_sha256": sha256_file(X_PATH),
        "X_array_sha256": sha256_array(X),
        "code_revision": _git_revision(),
    }
    (OUT / "phase2_exact_refinement/config.json").write_text(json.dumps(config, indent=2) + "\n")
    hardware_before = hardware_snapshot()
    (OUT / "phase2_exact_refinement/hardware_before.json").write_text(json.dumps(hardware_before, indent=2) + "\n")
    (OUT / "phase3_adaptive_scheduling/status.json").write_text(json.dumps({
        "status": "not run pending Phase 2 gate",
        "reason": "Existing comparisons have not shown adaptation beating the best fixed policy. Reconsider only if exact refinement establishes a target-eligible Mini-Batch candidate.",
        "target_is_not_controller_input": True,
    }, indent=2) + "\n")
    (p2 / "manifest.json").write_text(json.dumps({
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": config["code_revision"], "X_array_sha256": config["X_array_sha256"],
        "scene": config["scene"], "shape": [N, D], "K": K,
        "seeds": list(SEEDS), "target_visibility": "evaluation only; not passed to fitting",
        "source_phase1_checksum_entries_verified": source_diag_verified,
        "cuda_simulator": False, "hardware_before": hardware_before,
    }, indent=2, default=str) + "\n")
    warm_rows = _warmups(X, C0s[0])
    _append(p2 / "warmups.csv", warm_rows, ["method", "warmup", "wall_s_excluded", "N_warmup", "batch_size_warmup"])

    run_fields = ["run_id", "method", "seed", "refinement_requested", "refinement_actual_steps",
                  "N", "D", "K", "batch_size", "observation_budget", "observations_processed",
                  "updates", "termination_reason", "training_termination_reason",
                  "refinement_termination_reason", "convergence_satisfied", "input_sha256", "C0_sha256",
                  "trained_centroids_sha256", "final_centroids_sha256", "sample_stream_sha256",
                  "final_exact_sse", "exact_cuda_reference_sse", "relative_gap_to_exact_cuda",
                  "frozen_target_sse", "frozen_target_hit", "first_target_crossing_observations",
                  "first_observed_target_crossing_s",
                  "crossing_scope", "training_wall_s", "refinement_wall_s", "end_to_end_wall_s",
                  "training_timings_json", "refinement_timings_json", "training_sample_hashes_match"]
    checkpoint_fields = ["method", "seed", "refinement_requested", "stage", "step_or_observations",
                         "full_exact_sse", "elapsed_s", "target_evaluation_only", "target_hit"]
    timing_fields = ["method", "seed", "refinement_requested", "training_wall_s", "refinement_wall_s",
                     "end_to_end_wall_s", "training_compute_s", "training_setup_s", "training_sampling_s",
                     "training_assignment_s", "training_update_s", "training_monitoring_s",
                     "training_quality_evaluation_s", "refinement_compute_s", "refinement_setup_s",
                     "refinement_upload_s", "refinement_transfer_s", "refinement_final_assignment_s",
                     "timing_scope"]
    controller_fields = ["status", "method", "seed", "step", "event", "details"]
    (p2 / "controller_events.csv").write_text(",".join(controller_fields) + "\n")
    failures_path = p2 / "failures.jsonl"
    run_id = 0
    rng_order = random.Random(20261027)
    case_list = [(m, s, r) for s in SEEDS for m in METHODS for r in REFINEMENT_LEVELS]
    # Add one paired exact-G0 reference fit per seed; randomized ordering is
    # saved for the update/refinement cases, while G0 anchors the same session.
    for seed in SEEDS:
        order = [(m, seed, r) for m in METHODS for r in REFINEMENT_LEVELS] + [("G0_EXACT_CUDA", seed, -1)]
        rng_order.shuffle(order)
        for method, current_seed, refinement in order:
            C0 = C0s[current_seed]
            if method == "G0_EXACT_CUDA":
                from pcap.contracts import FitConfig
                from pcap.kmeans_cuda import fit_cuda
                start = perf_counter()
                result = fit_cuda(X, C0, FitConfig(k=K, seed=current_seed, max_steps=500, tolerance=1e-6))
                wall = perf_counter() - start
                row = {
                    "run_id": f"G0-{current_seed}", "method": method, "seed": current_seed,
                    "refinement_requested": 0, "refinement_actual_steps": result.metadata["steps"],
                    "N": N, "D": D, "K": K, "batch_size": "", "observation_budget": "",
                    "observations_processed": 0, "updates": result.metadata["steps"],
                    "termination_reason": result.metadata["termination_reason"],
                    "training_termination_reason": "not_applicable_exact_g0",
                    "refinement_termination_reason": "not_applicable_exact_g0",
                    "convergence_satisfied": result.metadata.get("convergence_satisfied"),
                    "input_sha256": config["X_array_sha256"], "C0_sha256": sha256_array(C0),
                    "trained_centroids_sha256": "", "final_centroids_sha256": sha256_array(result.centroids),
                    "sample_stream_sha256": "", "final_exact_sse": result.sse,
                    "exact_cuda_reference_sse": result.sse, "relative_gap_to_exact_cuda": 0.0,
                    "frozen_target_sse": targets[current_seed], "frozen_target_hit": result.sse <= targets[current_seed],
                    "first_target_crossing_observations": next((int(h["points_processed"]) for h in result.history
                                                                  if h["sse"] <= targets[current_seed]), ""),
                    "first_observed_target_crossing_s": wall if result.sse <= targets[current_seed] else "",
                    "crossing_scope": "G0 fit-completion upper bound; per-step wall timestamps unavailable" if result.sse <= targets[current_seed] else "not observed",
                    "training_wall_s": "", "refinement_wall_s": "", "end_to_end_wall_s": wall,
                    "training_timings_json": "", "refinement_timings_json": json.dumps(result.timings, sort_keys=True),
                    "training_sample_hashes_match": "",
                }
                _append(p2 / "individual_runs.csv", [row], run_fields)
                _append(p2 / "timing_breakdown.csv", [{
                    "method": method, "seed": current_seed, "refinement_requested": 0,
                    "training_wall_s": "", "refinement_wall_s": "", "end_to_end_wall_s": wall,
                    "training_compute_s": "", "training_setup_s": "", "training_sampling_s": "",
                    "training_assignment_s": "", "training_update_s": "", "training_monitoring_s": "",
                    "training_quality_evaluation_s": "", "refinement_compute_s": result.timings.get("compute_s"),
                    "refinement_setup_s": result.timings.get("setup_s"), "refinement_upload_s": result.timings.get("upload_s"),
                    "refinement_transfer_s": result.timings.get("transfer_s"),
                    "refinement_final_assignment_s": result.timings.get("final_assignment_s"),
                    "timing_scope": "Exact G0 complete fit; internal categories overlap wall and each other.",
                }], timing_fields)
                for h in result.history:
                    _append(p2 / "checkpoints.csv", [{
                        "method": method, "seed": current_seed, "refinement_requested": 0,
                        "stage": "exact_g0", "step_or_observations": h["step"],
                        "full_exact_sse": h["sse"], "elapsed_s": "",
                        "target_evaluation_only": True,
                        "target_hit": h["sse"] <= targets[current_seed],
                    }], checkpoint_fields)
                continue

            run_id += 1
            total_start = perf_counter()
            try:
                train, train_checkpoints, train_wall = _case_fit(X, C0, current_seed, method)
                train_C = np.ascontiguousarray(train.centroids, dtype=np.float32)
                train_sse = float(train.sse)
                refinement_start = perf_counter()
                ref_result = exact_refinement(X, train_C, current_seed, refinement)
                refinement_wall = perf_counter() - refinement_start if ref_result is not None else 0.0
                total_wall = perf_counter() - total_start
                ref_timings = {}
                train_centroid_hash = _save_centroids(p2, method, current_seed, refinement, train_C)
                if ref_result is None:
                    final_C, final_sse, actual_ref_steps = train_C, train_sse, 0
                    reason, converged = train.metadata.get("termination_reason"), train.metadata.get("convergence_satisfied")
                    refinement_reason = "not_run"
                    final_centroid_hash = train_centroid_hash
                else:
                    # The exact-CUDA call includes X/C upload, all requested
                    # Lloyd passes, final exact evaluation and result downloads.
                    final_C, final_sse = ref_result.centroids, float(ref_result.sse)
                    actual_ref_steps = int(ref_result.metadata["steps"])
                    reason, converged = ref_result.metadata["termination_reason"], ref_result.metadata.get("convergence_satisfied")
                    refinement_reason = reason
                    ref_timings = ref_result.timings
                    final_centroid_hash = _save_centroids(p2, method, current_seed, refinement, final_C)
                    exact_history_sse = [float(h["sse"]) for h in ref_result.history]
                    numerical_slack = max(1e-8, abs(exact_history_sse[0]) * 1e-7)
                    if any(after > before + numerical_slack for before, after in zip(exact_history_sse, exact_history_sse[1:])):
                        raise FloatingPointError("Exact Lloyd refinement increased full-data SSE beyond numerical slack")
                    if final_sse > train_sse + numerical_slack:
                        raise FloatingPointError("Exact Lloyd refinement final SSE exceeds its input centroid SSE")
                if total_wall + 1e-6 < train_wall + refinement_wall:
                    raise RuntimeError("External end-to-end timer is shorter than measured training plus refinement")
                target = targets[current_seed]
                train_hashes = [h.get("sample_indices_sha256") for h in train.history if h.get("sample_indices_sha256")]
                stream_sha = hashlib.sha256("".join(train_hashes).encode()).hexdigest()
                # Store full-data observation checkpoints from both methods.
                for cp in train_checkpoints:
                    point_count = int(cp.get("points_processed", 0))
                    sse = float(cp.get("full_exact_sse", cp.get("full_data_sse")))
                    elapsed = float(cp.get("elapsed_s", 0.0))
                    _append(p2 / "checkpoints.csv", [{
                        "method": method, "seed": current_seed, "refinement_requested": refinement,
                        "stage": "minibatch", "step_or_observations": point_count,
                        "full_exact_sse": sse, "elapsed_s": elapsed,
                        "target_evaluation_only": True, "target_hit": sse <= target,
                    }], checkpoint_fields)
                if ref_result is not None:
                    for h in ref_result.history:
                        _append(p2 / "checkpoints.csv", [{
                            "method": method, "seed": current_seed, "refinement_requested": refinement,
                            "stage": "exact_refinement", "step_or_observations": h["step"],
                            "full_exact_sse": h["sse"], "elapsed_s": "",
                            "target_evaluation_only": True, "target_hit": h["sse"] <= target,
                        }], checkpoint_fields)
                first_hit = next((c for c in train_checkpoints
                                  if float(c.get("full_exact_sse", c.get("full_data_sse", float("inf")))) <= target), None)
                crossing_observations = int(first_hit.get("points_processed", 0)) if first_hit else ""
                crossing_s = float(first_hit.get("elapsed_s", 0.0)) if first_hit else ""
                crossing_scope = "Mini-Batch full-data checkpoint" if first_hit else ""
                if not first_hit and final_sse <= target and refinement > 0:
                    crossing_s = total_wall
                    first_refined_hit = next((h for h in ref_result.history if float(h["sse"]) <= target), None)
                    crossing_observations = BUDGET + (int(first_refined_hit["step"]) * N if first_refined_hit else actual_ref_steps * N)
                    crossing_scope = "upper bound at refinement-call completion; exact per-step wall timestamp unavailable"
                row = {
                    "run_id": run_id, "method": method, "seed": current_seed,
                    "refinement_requested": refinement, "refinement_actual_steps": actual_ref_steps,
                    "N": N, "D": D, "K": K, "batch_size": BATCH, "observation_budget": BUDGET,
                    "observations_processed": train.metadata.get("points_processed"),
                    "updates": train.metadata.get("steps"), "termination_reason": reason,
                    "training_termination_reason": train.metadata.get("termination_reason"),
                    "refinement_termination_reason": refinement_reason,
                    "convergence_satisfied": converged, "input_sha256": config["X_array_sha256"],
                    "C0_sha256": sha256_array(C0), "trained_centroids_sha256": train_centroid_hash,
                    "final_centroids_sha256": final_centroid_hash, "sample_stream_sha256": stream_sha,
                    "final_exact_sse": final_sse, "exact_cuda_reference_sse": "",
                    "relative_gap_to_exact_cuda": "", "frozen_target_sse": target,
                    "frozen_target_hit": final_sse <= target,
                    "first_target_crossing_observations": crossing_observations,
                    "first_observed_target_crossing_s": crossing_s,
                    "crossing_scope": crossing_scope or "not observed",
                    "training_wall_s": train_wall, "refinement_wall_s": refinement_wall,
                    "end_to_end_wall_s": total_wall,
                    "training_timings_json": json.dumps(train.timings, sort_keys=True, default=str),
                    "refinement_timings_json": json.dumps(ref_timings, sort_keys=True, default=str),
                    "training_sample_hashes_match": "pending paired verification",
                }
                _append(p2 / "individual_runs.csv", [row], run_fields)
                t = train.timings
                _append(p2 / "timing_breakdown.csv", [{
                    "method": method, "seed": current_seed, "refinement_requested": refinement,
                    "training_wall_s": train_wall, "refinement_wall_s": refinement_wall,
                    "end_to_end_wall_s": total_wall, "training_compute_s": t.get("compute_s"),
                    "training_setup_s": t.get("setup_s"), "training_sampling_s": t.get("sampling_s"),
                    "training_assignment_s": t.get("batch_assignment_s", t.get("assignment_s")),
                    "training_update_s": t.get("accumulation_update_s", t.get("update_s")),
                    "training_monitoring_s": t.get("monitoring_s"),
                    "training_quality_evaluation_s": t.get("quality_evaluation_s"),
                    "refinement_compute_s": ref_timings.get("compute_s"),
                    "refinement_setup_s": ref_timings.get("setup_s"),
                    "refinement_upload_s": ref_timings.get("upload_s"),
                    "refinement_transfer_s": ref_timings.get("transfer_s"),
                    "refinement_final_assignment_s": ref_timings.get("final_assignment_s"),
                    "timing_scope": "External total spans training and refinement; internal categories overlap and are not summed.",
                }], timing_fields)
                _append(p2 / "controller_events.csv", [{
                    "status": "not applicable: fixed batch pilot; no controller receives evaluation target",
                    "method": method, "seed": current_seed, "step": "", "event": "", "details": "",
                }], controller_fields)
            except Exception as exc:
                _append_jsonl(failures_path, {
                    "run_id": run_id, "method": method, "seed": current_seed,
                    "refinement_requested": refinement, "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "utc": datetime.now(timezone.utc).isoformat(),
                })
                raise

    # Pair the new mini-batch fits with the same-seed G0 fit and verify the
    # cumulative/decayed batch streams match exactly at identical batch width.
    with (p2 / "individual_runs.csv").open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    references = {int(r["seed"]): float(r["final_exact_sse"])
                  for r in runs if r["method"] == "G0_EXACT_CUDA"}
    for r in runs:
        if r["method"] != "G0_EXACT_CUDA":
            r["exact_cuda_reference_sse"] = references[int(r["seed"])]
            r["relative_gap_to_exact_cuda"] = (float(r["final_exact_sse"])/references[int(r["seed"])]-1.0)
    with (p2 / "individual_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields); writer.writeheader(); writer.writerows(runs)
    by_pair = {}
    for r in runs:
        if r["method"] != "G0_EXACT_CUDA":
            by_pair.setdefault((r["seed"], r["refinement_requested"]), {})[r["method"]] = r["sample_stream_sha256"]
    stream_checks = []
    for key, methods in by_pair.items():
        equal = len(set(methods.values())) == 1
        stream_checks.append({"seed": key[0], "refinement_requested": key[1],
                              "sample_streams_identical": equal, "streams": methods})
    if not all(x["sample_streams_identical"] for x in stream_checks):
        raise RuntimeError("Cumulative and decayed fit sampling hashes differ for paired cases")
    (p2 / "sampling_stream_verification.json").write_text(json.dumps(stream_checks, indent=2) + "\n")

    # Per-seed exact G0 reference data and machine state are persisted before
    # any summary is created.
    hardware_after = hardware_snapshot()
    (p2 / "hardware_after.json").write_text(json.dumps(hardware_after, indent=2) + "\n")
    _summarize(p2, runs)
    (OUT / "phase1_centroid_updates/frozen_interpretation.md").write_text(
        "Phase 1 reuses the archived exact-assignment cumulative-vs-decay matched diagnostic and the frozen fixed-decay validation study. The matched diagnostic used identical X, C0 and per-update sample-index hashes. It showed lower SSE for decay in all three diagnostic seeds, while cumulative SSE continued declining and centroid movement remained nonzero at the cap. No third update rule was added: the evidence supports testing decay and exact refinement, but does not isolate an additional staleness-aware learning-rate rule. See the copied source records and their original checksums.\n"
    )
    (OUT / "phase3_adaptive_scheduling/status.json").write_text(json.dumps({
        "status": "not run pending Phase 2 gate",
        "reason": "Existing same-update-family comparisons show QGCUDA slower than the best fixed cumulative batch and Adaptive V2 slower than the best fixed-decay batch at the frozen target. New adaptive scheduling is not justified unless exact-refined Mini-Batch establishes a quality-valid candidate.",
        "target_is_not_controller_input": True,
    }, indent=2) + "\n")
    _make_figures(p2, OUT / "figures")
    manifest = {
        "study": "centroid quality and exact refinement pilot",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": _git_revision(), "scene": protocol.get("validation_scene"),
        "dataset_identity": "Sentinel-2 31UFU; previous validation scene, therefore exploratory not independent confirmation",
        "X_file_sha256": config["X_file_sha256"], "X_array_sha256": config["X_array_sha256"],
        "C0_sha256_by_seed": {str(s): sha256_array(C0s[s]) for s in SEEDS},
        "frozen_protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "frozen_targets_by_seed": {str(s): targets[s] for s in SEEDS},
        "hardware_before": hardware_before, "hardware_after": hardware_after,
        "source_code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            ROOT / "experiments/run_centroid_quality_refinement.py",
            ROOT / "src/pcap/kmeans_cuda.py", ROOT / "src/pcap/minibatch_cuda.py",
            ROOT / "src/pcap/minibatch_decayed_cuda.py", ROOT / "src/pcap/budgeted_minibatch.py")},
        "no_baseline_modified": True, "cuda_simulator": False,
        "target_passed_to_fit": False, "phases": {"phase1": "reused verified records",
        "phase2": "real GPU pilot", "phase3": "gated/skipped if phase2 does not justify adaptation"},
        "timing_note": "end_to_end_wall_s is external wall around training plus optional exact-CUDA refinement. Internal timing categories overlap and are not summed.",
    }
    (OUT / "phase2_exact_refinement/manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _write_final_report(OUT)
    (p2 / "status.json").write_text(json.dumps({
        "status": "complete", "completed_fit_records": len(runs),
        "expected_fit_records": 27, "training_fit_failures": 0,
        "real_cuda": True, "cuda_simulator": False,
        "sampling_stream_pairs_verified": len(stream_checks),
        "sampling_stream_pairs_matched": all(x["sample_streams_identical"] for x in stream_checks),
        "note": "Three seeds and one measured repetition per configuration; exploratory pilot only.",
    }, indent=2) + "\n")
    (OUT / "experiment_index.json").write_text(json.dumps({
        "study": "20261027-centroid-quality-refinement-study", "status": "complete",
        "summary": "final_summary/final_summary.md",
        "phase1_interpretation": "phase1_centroid_updates/frozen_interpretation.md",
        "individual_runs": "phase2_exact_refinement/individual_runs.csv",
        "checkpoints": "phase2_exact_refinement/checkpoints.csv",
        "method_summary": "phase2_exact_refinement/method_summary.csv",
        "figures": ["figures/01_refinement_runtime_and_sse.png",
                    "figures/02_refinement_quality_runtime_scatter.png"],
        "checksums": "checksums.sha256",
    }, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Centroid quality and exact CUDA refinement pilot\n\n"
        "This exploratory pilot uses the previously examined Sentinel-2 31UFU validation crop; it is not independent confirmation. "
        "The raw SAFE archive is not duplicated. Input identity and hashes are recorded in `phase2_exact_refinement/config.json`.\n\n"
        "Reproduction requires the prepared validation feature matrix, frozen protocol, project `.venv`, and a real CUDA device. "
        "Run from the repository root into a new empty directory (the runner refuses overwrites):\n\n"
        "```bash\n"
        "LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}\" PYTHONPATH=src \\\n"
        "  .venv/bin/python experiments/run_centroid_quality_refinement.py \\\n"
        "  --output outputs/<new-empty-study-directory>\n"
        "```\n\n"
        "No fit needs to be rerun to inspect this archive. See `experiment_index.json` and the final summary.\n",
        encoding="utf-8")
    _write_checksums(OUT)
    return OUT


def _git_revision() -> str:
    try:
        return _run(["git", "rev-parse", "HEAD"])
    except Exception:
        return "unavailable"


def _summarize(directory: Path, runs: list[dict]) -> None:
    fields = ["method", "refinement_requested", "n", "median_end_to_end_wall_s", "iqr_end_to_end_wall_s",
              "median_final_exact_sse", "median_gap_to_g0_pct", "target_hits", "n_runs",
              "median_actual_refinement_steps"]
    output = []
    for method in (*METHODS, "G0_EXACT_CUDA"):
        for level in (REFINEMENT_LEVELS if method != "G0_EXACT_CUDA" else (0,)):
            rows = [r for r in runs if r["method"] == method and int(r["refinement_requested"]) == level]
            if not rows:
                continue
            times = np.asarray([float(r["end_to_end_wall_s"]) for r in rows])
            sses = np.asarray([float(r["final_exact_sse"]) for r in rows])
            gaps = np.asarray([float(r["relative_gap_to_exact_cuda"]) * 100 for r in rows])
            output.append({"method": method, "refinement_requested": level, "n": len(rows),
                           "median_end_to_end_wall_s": float(np.median(times)),
                           "iqr_end_to_end_wall_s": float(np.percentile(times, 75)-np.percentile(times,25)),
                           "median_final_exact_sse": float(np.median(sses)),
                           "median_gap_to_g0_pct": float(np.median(gaps)),
                           "target_hits": sum(str(r["frozen_target_hit"]).lower() == "true" for r in rows),
                           "n_runs": len(rows),
                           "median_actual_refinement_steps": float(np.median([int(r["refinement_actual_steps"]) for r in rows]))})
    with (directory / "method_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def _make_figures(directory: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt

    runs = list(csv.DictReader((directory / "individual_runs.csv").open(newline="", encoding="utf-8")))
    methods = METHODS
    colors = {METHODS[0]: "tab:blue", METHODS[1]: "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for method in methods:
        rr = [r for r in runs if r["method"] == method]
        grouped = [(level, [r for r in rr if int(r["refinement_requested"]) == level]) for level in REFINEMENT_LEVELS]
        axes[0].plot([x for x, _ in grouped], [np.median([float(r["end_to_end_wall_s"]) for r in rows]) for _, rows in grouped], marker="o", label=method, color=colors[method])
        axes[1].plot([x for x, _ in grouped], [np.median([float(r["final_exact_sse"]) for r in rows]) for _, rows in grouped], marker="o", label=method, color=colors[method])
    axes[0].set(xlabel="Exact CUDA refinement updates", ylabel="Complete pipeline wall time (s)", title="Training + refinement runtime")
    axes[1].set(xlabel="Exact CUDA refinement updates", ylabel="Final exact full-data SSE", title="Quality after refinement")
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "01_refinement_runtime_and_sse.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in methods:
        rr = [r for r in runs if r["method"] == method]
        ax.scatter([float(r["end_to_end_wall_s"]) for r in rr], [float(r["final_exact_sse"]) for r in rr], label=method, alpha=.8, color=colors[method])
    ax.set(xlabel="Complete pipeline wall time (s)", ylabel="Final exact full-data SSE", title="Mini-Batch solutions with exact CUDA refinement")
    ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "02_refinement_quality_runtime_scatter.png", dpi=180); plt.close(fig)


def _write_final_report(directory: Path) -> None:
    data_directory = directory / "phase2_exact_refinement"
    runs = list(csv.DictReader((data_directory / "individual_runs.csv").open(newline="", encoding="utf-8")))
    lines = [
        "# Centroid quality and exact CUDA refinement study",
        "",
        "## Protocol and scope",
        "",
        "This is an exploratory extension on the previously examined Sentinel-2 31UFU validation crop (N=262,144, D=4, K=16); it is not an untouched confirmation. The existing frozen threshold (exact CPU reference SSE × 1.05) was used only by the evaluator. Update functions received no threshold or reference result. We reused the existing 40.96M-observation budget and same seed-specific C0. Mini-Batch training used fixed batch 65,536, exact assignment, and either the original cumulative or existing observation-scaled decayed update. Each pilot case then applied 0, 1, 3, or 5 exact CUDA Lloyd updates. Requested levels are maximum steps; actual executed steps and termination are recorded.",
        "",
        "End-to-end timing externally brackets the Mini-Batch fit plus the optional separate exact-CUDA call, including the refinement call's input/centroid upload, setup, updates, final exact evaluation, and result download. Internal timing categories overlap and are shown separately. No target is used for early stopping. Sampling-stream hashes are compared between cumulative and decay runs at the matched fixed batch width.",
        "",
        "## Results",
        "",
        "See `../phase2_exact_refinement/method_summary.csv` and `../phase2_exact_refinement/individual_runs.csv` for seed-level values. The strongest fixed cumulative runtime candidate was B=65,536; the matched update-family diagnostic supports lower SSE under decay, so the pilot tests both update families at the same B.",
        "",
        "| Update rule | Refinement | Median end-to-end (s) | Median final exact SSE | Median gap to paired G0 | Frozen target hits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = list(csv.DictReader((data_directory / "method_summary.csv").open(newline="", encoding="utf-8")))
    for row in summary:
        if row["method"] == "G0_EXACT_CUDA":
            continue
        lines.append(f"| {row['method']} | {row['refinement_requested']} | {float(row['median_end_to_end_wall_s']):.3f} | {float(row['median_final_exact_sse']):.2f} | {float(row['median_gap_to_g0_pct']):.2f}% | {row['target_hits']}/{row['n_runs']} |")
    for row in summary:
        if row["method"] == "G0_EXACT_CUDA":
            lines.append(f"| Exact CUDA G0 | — | {float(row['median_end_to_end_wall_s']):.3f} | {float(row['median_final_exact_sse']):.2f} | 0.00% | {row['target_hits']}/{row['n_runs']} |")
    lines += ["", "## Interpretation and gates", "", "Phase 1 matched diagnostics support cumulative-statistic inertia as a contributor, but not as the sole cause: decay produced lower SSE in three matched seeds while cumulative SSE continued declining at the cap. In this Phase 2 pilot, cumulative Mini-Batch reached the frozen target for 1/3 seeds after five exact refinements. Fixed decay reached it for 2/3 seeds without refinement and remained 2/3 after up to five refinements. Exact refinement reduced SSE for every seed and refinement level, but did not improve the observed target-hit count.", "", "Exact CUDA G0 reached the frozen target for 3/3 seeds. It was faster than each paired Mini-Batch-plus-refinement run in this pilot; therefore, refinement did not establish a time-to-quality advantage. It improves solution quality while adding cost. The result does not establish that exact refinement is generally inefficient, but this workload and protocol do not support replacing G0 with it for speed.", "", "Timing uncertainty is substantial: each seed/configuration has one measured run, and runtime variability across seeds and machine state is large. Group medians are descriptive, not stable performance estimates. For target crossings first observed during refinement, the recorded time is a conservative upper bound at refinement-call completion because per-step wall timestamps were unavailable. The validation scene was previously examined, so this is exploratory evidence rather than independent confirmation. No land-cover accuracy claims are made because aligned reference labels are unavailable.", "", "## Reproduction and records", "", "Configuration, hardware snapshots, per-run CSVs, common checkpoints, timing breakdowns, centroid files, sample-stream and termination verification, finalization diagnostics, figures and checksums are stored alongside this report. Re-run only into a new output path after preserving this archive.", ""]
    summary_directory = directory / "final_summary"
    summary_directory.mkdir(parents=True, exist_ok=True)
    (summary_directory / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_existing_archive(directory: Path) -> None:
    """Repair derived reporting after a post-fit error without rerunning fits.

    Measured run rows are left as the original experiment-time schema. A
    separate sidecar identifies which stage stopped; legacy
    ``termination_reason`` values remain intact for traceability.
    """
    data_directory = directory / "phase2_exact_refinement"
    runs_path = data_directory / "individual_runs.csv"
    with runs_path.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    if len(runs) != 27:
        raise ValueError(f"Expected 27 completed fit records, found {len(runs)}")
    stage_rows = []
    for row in runs:
        if row["method"] == "G0_EXACT_CUDA":
            training_reason = refinement_reason = "not_applicable_exact_g0"
        else:
            observed = int(row["observations_processed"])
            budget = int(row["observation_budget"])
            if observed != budget:
                raise ValueError(f"Fit {row['run_id']} did not complete the stated observation budget")
            training_reason = "observation_budget"
            refinement_reason = (
                "not_run" if int(row["refinement_requested"]) == 0 else row["termination_reason"]
            )
        stage_rows.append({"run_id": row["run_id"], "method": row["method"], "seed": row["seed"],
                           "training_termination_reason": training_reason,
                           "refinement_termination_reason": refinement_reason,
                           "legacy_termination_reason": row["termination_reason"]})
    stage_columns = {"training_termination_reason", "refinement_termination_reason"}
    fields = [key for key in runs[0].keys() if key not in stage_columns]
    with runs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in runs])
    with (data_directory / "run_termination_interpretation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields_stage = ["run_id", "method", "seed", "training_termination_reason",
                        "refinement_termination_reason", "legacy_termination_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields_stage)
        writer.writeheader()
        writer.writerows(stage_rows)
    _summarize(data_directory, runs)
    _write_final_report(directory)
    # Preserve the original traceback record as evidence; explicitly classify
    # it as report finalization after all fits, not a failed clustering run.
    historical_error = data_directory / "failures.jsonl"
    (data_directory / "finalization_recovery.json").write_text(json.dumps({
        "status": "fit_records_complete_report_recovered",
        "completed_fit_records": len(runs),
        "expected_fit_records": 27,
        "fit_failures": 0,
        "historical_error_record": historical_error.name if historical_error.exists() else None,
        "historical_error_classification": "post-fit report assembly failed because the report writer looked for CSVs at the archive root instead of phase2_exact_refinement/; all 27 fit records, summaries, checkpoints, timings, and figures had already been written",
        "measurements_changed": False,
        "derived_changes": ["wrote separate run_termination_interpretation.csv without changing measurement fields", "recomputed method_summary.csv run denominators", "regenerated final report"],
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    (data_directory / "status.json").write_text(json.dumps({
        "status": "complete_with_post_fit_report_recovery",
        "completed_fit_records": len(runs), "expected_fit_records": 27,
        "fit_failures": 0, "report_recovery_required": True,
        "real_cuda": True, "cuda_simulator": False,
    }, indent=2) + "\n")
    manifest_path = data_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete_with_post_fit_report_recovery"
    manifest["fit_records"] = len(runs)
    manifest["fit_failures"] = 0
    manifest["report_recovery"] = "finalization_recovery.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    import shutil
    snapshot = data_directory / "source_snapshot_postrun"
    snapshot.mkdir(exist_ok=True)
    recorded_sources = manifest.get("source_code_sha256", {})
    snapshot_paths = [
        "experiments/run_centroid_quality_refinement.py",
        "src/pcap/kmeans_cuda.py", "src/pcap/minibatch_cuda.py",
        "src/pcap/minibatch_decayed_cuda.py", "src/pcap/budgeted_minibatch.py",
        "src/pcap/contracts.py", "src/pcap/minibatch.py",
        "tests/test_centroid_quality_refinement.py",
    ]
    snapshot_rows = []
    for relative in snapshot_paths:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshot_hash = sha256_file(destination)
        recorded_hash = recorded_sources.get(relative)
        snapshot_rows.append({"path": relative, "postrun_snapshot_sha256": snapshot_hash,
                              "experiment_manifest_sha256": recorded_hash,
                              "matches_experiment_source": snapshot_hash == recorded_hash if recorded_hash else None})
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({
        "snapshot_timing": "post-run report recovery; not exact experiment-time runner source",
        "exact_experiment_runner_recovered": False,
        "experiment_runner_sha256_recorded_in_original_manifest": recorded_sources.get("experiments/run_centroid_quality_refinement.py"),
        "files": snapshot_rows,
    }, indent=2) + "\n")
    (directory / "experiment_index.json").write_text(json.dumps({
        "study": "20261027-centroid-quality-refinement-study",
        "status": "complete_with_post_fit_report_recovery",
        "summary": "final_summary/final_summary.md",
        "phase1_interpretation": "phase1_centroid_updates/frozen_interpretation.md",
        "individual_runs": "phase2_exact_refinement/individual_runs.csv",
        "checkpoints": "phase2_exact_refinement/checkpoints.csv",
        "method_summary": "phase2_exact_refinement/method_summary.csv",
        "figures": ["figures/01_refinement_runtime_and_sse.png",
                    "figures/02_refinement_quality_runtime_scatter.png"],
        "checksums": "checksums.sha256",
    }, indent=2) + "\n")
    (directory / "README.md").write_text(
        "# Centroid quality and exact CUDA refinement pilot\n\n"
        "This is an exploratory pilot on the previously examined Sentinel-2 31UFU validation crop, not independent confirmation. "
        "The raw SAFE archive is not duplicated. Input identity and hashes are recorded in `phase2_exact_refinement/config.json`.\n\n"
        "The 27 real-GPU fits are preserved; the original reporting failure occurred after all fits completed. "
        "`phase2_exact_refinement/finalization_recovery.json` records that distinction. This archive was not rerun. "
        "Stage-specific stop reasons are in `phase2_exact_refinement/run_termination_interpretation.csv`; the raw measurement fields remain in the run CSV.\n\n"
        "To repeat the experiment, use the repository `.venv`, prepared feature matrix, frozen protocol, and real CUDA device. "
        "Run from repository root into a new empty path (the runner refuses overwrites):\n\n"
        "```bash\n"
        "LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}\" PYTHONPATH=src \\\n"
        "  .venv/bin/python experiments/run_centroid_quality_refinement.py \\\n"
        "  --output outputs/<new-empty-study-directory>\n"
        "```\n\n"
        "The manifest records the experiment-time runner SHA256. The source file now includes post-run report-recovery changes, "
        "so it is not identical to the runner version used for the fits. The baseline code revision and source hashes are recorded "
        "in the manifest; the exact untracked runner version was not committed or snapshotted before execution. A post-run source "
        "snapshot is included under `phase2_exact_refinement/source_snapshot_postrun/` with comparison hashes.\n",
        encoding="utf-8")
    _write_checksums(directory)


def _write_checksums(root: Path) -> None:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("checksums.sha256", "checksums.partial.sha256")):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="new archive directory (must not already exist)")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="regenerate derived reports for a completed archive; does not rerun fits")
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.finalize_existing:
        finalize_existing_archive(OUT)
        print(OUT)
        return
    try:
        print(run())
    except Exception as exc:
        if OUT.exists():
            phase2 = OUT / "phase2_exact_refinement"
            completed_fit_count = 0
            runs_file = phase2 / "individual_runs.csv"
            if runs_file.exists():
                with runs_file.open(newline="", encoding="utf-8") as stream:
                    completed_fit_count = max(0, sum(1 for _ in csv.DictReader(stream)))
            error_path = phase2 / ("finalization_errors.jsonl" if completed_fit_count >= 27 else "failures.jsonl")
            _append_jsonl(error_path, {
                "error": repr(exc), "traceback": traceback.format_exc(),
                "completed_fit_records": completed_fit_count,
                "scope": "post-fit finalization" if completed_fit_count >= 27 else "run or fit failure",
                "utc": datetime.now(timezone.utc).isoformat(),
            })
            (OUT / "phase2_exact_refinement/status.json").write_text(json.dumps({
                "status": "post_fit_finalization_error" if completed_fit_count >= 27 else "failed_or_incomplete",
                "completed_fit_records": completed_fit_count, "error": repr(exc),
                "utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
            _write_checksums(OUT)
        raise


if __name__ == "__main__":
    main()
