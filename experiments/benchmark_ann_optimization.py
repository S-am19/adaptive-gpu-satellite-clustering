"""Controlled real-scene feasibility gate for GPU centroid-assignment ANN.

All reported assignment times are external synchronized wall times. ANN rows
charge initial index construction to the initial-centroid query and charge a
device-to-host centroid copy, CPU index rebuild, upload, query, and result
downloads to the moved-centroid query. The two rows can be summed for a paired
two-state index lifecycle. Internal query/build values are diagnostic subsets
and must not be summed into the external wall time.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numba import cuda
from pcap.assignments_ann_cuda import ProjectionCUDAIndex
from pcap.assignments_ann_multi_cuda import MultiProjectionCUDAIndex
from pcap.assignments_exact import assign_exact
from pcap.contracts import FitConfig
from pcap.initialization import initialize_centroids
from pcap.kmeans_cuda import fit_cuda
from pcap.minibatch_cuda import _assign_indexed_kernel, _blocks, _THREADS


SCENES = {
    "31UFS": ROOT / "outputs/20261024-plan-real-prepared/maastricht512",
    "31UFU": ROOT / "outputs/20261025-independent-holdouts/validation512",
}
K_VALUES = (16, 32, 64, 128, 256)
OLD_EFFORTS = (1, 4, 8, 16, 32)
MULTI_EFFORTS = (4, 8, 16, 32)
GATE = {"strict_agreement_min": 0.95, "assignment_objective_gap_max": 0.01,
        "median_paired_lifecycle_speedup_min": 1.10,
        "seed_bootstrap_speedup_ci95_lower_gt": 1.0,
        "both_centroid_states_required": True}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_array(array):
    return sha_bytes(np.ascontiguousarray(array).tobytes())


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nvidia_smi():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.free,memory.total,pstate,temperature.gpu,clocks.sm,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def package_versions():
    result = {}
    for name in ("numpy", "numba", "numba-cuda", "cuda-python", "llvmlite", "rasterio"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_hashes():
    paths = ["experiments/benchmark_ann_optimization.py",
             "src/pcap/assignments_ann_cuda.py",
             "src/pcap/assignments_ann_multi_cuda.py",
             "src/pcap/minibatch_cuda.py", "src/pcap/kmeans_cuda.py",
             "src/pcap/initialization.py", "src/pcap/assignments_exact.py"]
    return {p: sha_bytes((ROOT / p).read_bytes()) for p in paths}


def check_real_gpu():
    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError("CUDA simulator is enabled; real GPU execution is required")
    if not cuda.is_available():
        raise RuntimeError("No real CUDA device is available; CPU fallback is disabled")
    device = cuda.get_current_device()
    return {"device_name": device.name.decode() if isinstance(device.name, bytes) else str(device.name),
            "compute_capability": list(device.compute_capability),
            "memory_info_bytes": list(cuda.current_context().get_memory_info())}


def load_scene(scene_key):
    path = SCENES[scene_key]
    X = np.load(path / "X.npy", allow_pickle=False)
    metadata = json.loads((path / "metadata.json").read_text())
    if X.dtype != np.float32 or X.shape != (262144, 4) or not X.flags.c_contiguous:
        raise ValueError(f"Unexpected prepared scene shape/dtype/layout: {X.shape}/{X.dtype}")
    if not np.isfinite(X).all() or sha_array(X) != metadata["X_sha256"]:
        raise ValueError(f"Prepared feature matrix hash/finite validation failed for {scene_key}")
    return X, metadata


def _exact_query(X, C, resident_centroids=None):
    n = len(X)
    start = perf_counter()
    dx = cuda.to_device(X)
    dc = cuda.to_device(C) if resident_centroids is None else resident_centroids
    indices = cuda.to_device(np.arange(n, dtype=np.int32))
    labels = cuda.device_array(n, dtype=np.int32)
    distances = cuda.device_array(n, dtype=np.float64)
    cuda.synchronize()
    transfer_in_s = perf_counter() - start
    query_start = perf_counter()
    _assign_indexed_kernel[_blocks(n), _THREADS](dx, indices, dc, labels, distances)
    cuda.synchronize()
    query_s = perf_counter() - query_start
    copy_start = perf_counter()
    host_labels = labels.copy_to_host()
    host_distances = distances.copy_to_host()
    cuda.synchronize()
    copy_out_s = perf_counter() - copy_start
    return dict(labels=host_labels, distances=host_distances,
                external_s=perf_counter() - start, query_s=query_s,
                transfer_in_s=transfer_in_s, transfer_out_s=copy_out_s,
                index_build_s=0.0, index_refresh_s=0.0,
                candidate_evaluations=n * len(C), index_bytes=0)


def _ann_lifecycle(backend_cls, X, C0, C1, effort, seed):
    """Measure initial query, then refresh+query on the same resident index."""
    n = len(X)
    backend = backend_cls(seed=seed, ann_effort=effort)
    rows = []
    device_C1 = cuda.to_device(C1)  # mimics model state already resident after update

    # Initial build + point/index/output transfers + query + result downloads.
    start = perf_counter()
    backend.prepare(C0, effort)
    built = backend.timings
    device_X = cuda.to_device(X)
    device_indices = cuda.to_device(np.arange(n, dtype=np.int32))
    labels = cuda.device_array(n, dtype=np.int32)
    distances = cuda.device_array(n, dtype=np.float64)
    counts = cuda.device_array(n, dtype=np.int32) if backend_cls is MultiProjectionCUDAIndex else None
    cuda.synchronize()
    transfer_in = perf_counter() - start - built["initial_build_s"]
    if counts is None:
        result = backend.assign_device(device_X, device_indices, labels, distances,
                                       version=backend.version, effort=effort)
    else:
        result = backend.assign_device(device_X, device_indices, labels, distances, counts,
                                       version=backend.version, effort=effort)
    q_s = result.query_time_s
    copy_start = perf_counter()
    host_labels, host_distances = labels.copy_to_host(), distances.copy_to_host()
    cuda.synchronize()
    copy_out_s = perf_counter() - copy_start
    init_external = perf_counter() - start
    candidate_counts = counts.copy_to_host() if counts is not None else None
    cuda.synchronize()
    rows.append(dict(snapshot="initial", labels=host_labels, distances=host_distances,
        external_s=init_external, query_s=q_s, index_build_s=built["initial_build_s"],
        index_refresh_s=0.0, transfer_in_s=transfer_in, transfer_out_s=copy_out_s,
        candidate_evaluations=(int(candidate_counts.sum()) if candidate_counts is not None
                               else n * min(effort, len(C0))),
        index_bytes=getattr(backend, "index_bytes", getattr(backend, "device_index_bytes", 0))))

    # Charge the D2H centroid snapshot, CPU rebuild, re-upload, query, and
    # output downloads to the moved-centroid state. X/index I/O is reported as
    # distinct values when available; points/indices/outputs remain resident.
    refresh_start = perf_counter()
    backend.refresh_device(device_C1, version=backend.version + 1)
    refresh_s = perf_counter() - refresh_start
    device_X = cuda.to_device(X)
    device_indices = cuda.to_device(np.arange(n, dtype=np.int32))
    labels = cuda.device_array(n, dtype=np.int32)
    distances = cuda.device_array(n, dtype=np.float64)
    counts = cuda.device_array(n, dtype=np.int32) if backend_cls is MultiProjectionCUDAIndex else None
    cuda.synchronize()
    transfer_in = perf_counter() - refresh_start - refresh_s
    if counts is None:
        result = backend.assign_device(device_X, device_indices, labels, distances,
                                       version=backend.version, effort=effort)
    else:
        result = backend.assign_device(device_X, device_indices, labels, distances, counts,
                                       version=backend.version, effort=effort)
    q_s = result.query_time_s
    copy_start = perf_counter()
    host_labels, host_distances = labels.copy_to_host(), distances.copy_to_host()
    cuda.synchronize()
    copy_out_s = perf_counter() - copy_start
    moved_external = perf_counter() - refresh_start
    candidate_counts = counts.copy_to_host() if counts is not None else None
    cuda.synchronize()
    rows.append(dict(snapshot="after_one_exact_cuda_lloyd_update", labels=host_labels,
        distances=host_distances, external_s=moved_external, query_s=q_s,
        index_build_s=0.0, index_refresh_s=refresh_s, transfer_in_s=transfer_in,
        transfer_out_s=copy_out_s,
        candidate_evaluations=(int(candidate_counts.sum()) if candidate_counts is not None
                               else n * min(effort, len(C0))),
        index_bytes=getattr(backend, "index_bytes", getattr(backend, "device_index_bytes", 0))))
    return rows


def _warm(X, C0, C1, configs, seed, repeats=1):
    warm_X = np.ascontiguousarray(X[:min(8192, len(X))])
    for _ in range(repeats):
        for method, effort in configs:
            if method == "exact_cuda":
                _exact_query(warm_X, C0)
            else:
                cls = ProjectionCUDAIndex if method == "single_projection" else MultiProjectionCUDAIndex
                inst = cls(seed=seed, ann_effort=effort)
                inst.prepare(C0, effort)
                dx = cuda.to_device(warm_X)
                ids = cuda.to_device(np.arange(len(warm_X), dtype=np.int32))
                dl = cuda.device_array(len(warm_X), dtype=np.int32)
                dd = cuda.device_array(len(warm_X), dtype=np.float64)
                if method == "single_projection":
                    inst.assign_device(dx, ids, dl, dd, version=inst.version, effort=effort)
                else:
                    dc = cuda.device_array(len(warm_X), dtype=np.int32)
                    inst.assign_device(dx, ids, dl, dd, dc, version=inst.version, effort=effort)
                inst.refresh_device(cuda.to_device(C1), version=1)
    cuda.synchronize()


def _quality_row(labels, distances, exact_labels, exact_distances, X, C,
                 method, effort, scene, k, seed, rep, snapshot, x_hash, c_hash,
                 n, evals, external, query_s, build_s, refresh_s,
                 transfer_in_s, transfer_out_s, index_bytes):
    independent = np.sum((X.astype(np.float64) - C[labels].astype(np.float64)) ** 2,
                         axis=1, dtype=np.float64)
    np.testing.assert_allclose(distances, independent, rtol=1e-12, atol=1e-12)
    excess = np.maximum(distances - exact_distances, 0.0)
    exact_obj = float(exact_distances.sum(dtype=np.float64))
    obj = float(distances.sum(dtype=np.float64))
    return dict(scene=scene, n=n, d=4, k=k, seed=seed, repetition=rep,
        snapshot=snapshot, method=method, effort=effort,
        search_mode=("exact" if method == "exact_cuda" else
                     "single_projection_approximate" if method == "single_projection" else
                     ("exact_budget_fallback" if effort >= k else "multi_projection_approximate")),
        X_sha256=x_hash, centroids_sha256=c_hash,
        external_complete_s=external, query_s=query_s,
        index_build_s=build_s, index_refresh_s=refresh_s,
        transfer_in_s=transfer_in_s, transfer_out_s=transfer_out_s,
        strict_label_agreement=float(np.mean(labels == exact_labels)),
        tie_aware_distance_agreement=float(np.mean(np.isclose(
            distances, exact_distances, rtol=1e-10, atol=1e-12))),
        mean_excess_squared_distance=float(excess.mean()),
        max_excess_squared_distance=float(excess.max()),
        assignment_objective_gap=(obj - exact_obj) / max(abs(exact_obj), 1e-30),
        candidate_distance_evaluations=int(evals), exhaustive_distance_evaluations=n*k,
        candidate_fraction=float(evals/(n*k)), index_bytes=int(index_bytes),
        quality_status="measured exact-reference comparison")


def run(args):
    output = args.output_dir
    dirs = ["phaseA_ann_diagnostics", "phaseB_new_ann_backends",
            "phaseC_assignment_benchmarks", "phaseD_end_to_end_ann",
            "phaseE_combined_adaptation", "figures", "final_summary"]
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"cannot resume missing archive: {output}")
        for d in dirs:
            (output / d).mkdir(exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for d in dirs:
            (output / d).mkdir()
    gpu = check_real_gpu()
    seeds = args.seeds
    scenes = args.scenes
    protocol = dict(scenes=scenes, N=262144, D=4, K=list(K_VALUES), seeds=seeds,
        warmups_per_method=args.warmups, measured_repetitions=args.repeats,
        centroid_states=["seeded_initialization", "one_full_exact_CUDA_Lloyd_update"],
        single_projection_efforts=list(OLD_EFFORTS),
        multi_projection_efforts=list(MULTI_EFFORTS),
        candidate_efforts_filtered_to_effort_less_than_K=True,
        quality_gate=GATE, assignment_only=True,
        assignment_timings="synchronized external host wall; X and outputs transfer per query; initial C transfer/index build charged initially; post-update C state preloaded for both exact and ANN; ANN centroid D2H plus CPU rebuild/index upload charged on refresh; query is nested and not additive",
        exact_reference_validation_queries="two synchronized exact CUDA queries per case are correctness checks, excluded from raw measurement rows",
        centroid_init="initialize_centroids(X,K,seed); C1 from one fit_cuda Lloyd update with max_steps=1,tolerance=0",
        heldout_scene_30STG="not used")
    config = {"protocol": protocol, "pilot": bool(args.pilot),
              "time_budget_seconds": args.budget_seconds}
    config_path=output / "phaseC_assignment_benchmarks/config.json"
    manifest_path=output / "manifest.json"
    if args.resume:
        stored_config=json.loads(config_path.read_text())
        if stored_config.get("protocol")!=protocol or not stored_config.get("pilot"):
            raise ValueError("resume configuration differs from the incomplete pilot")
        meta=json.loads(manifest_path.read_text())
        current_hashes=source_hashes()
        previous_hashes=meta.get("source_hashes",{})
        changed=[p for p in set(previous_hashes)|set(current_hashes)
                 if previous_hashes.get(p)!=current_hashes.get(p)]
        if any(p!="experiments/benchmark_ann_optimization.py" for p in changed):
            raise ValueError(f"algorithm/protocol source changed since interruption: {changed}")
        meta["resume_runner_hash"] = current_hashes.get("experiments/benchmark_ann_optimization.py")
        meta["resume_note"] = "Only resume/append support in the runner changed after interruption; benchmark algorithms and measurement protocol source hashes were unchanged."
        if meta.get("complete"):
            raise ValueError("archive is already complete; refusing to resume")
        meta["previous_attempt_status"]="interrupted_before_manifest_finalization; completed CSV rows are retained and skipped on resume"
        meta.setdefault("resume_attempts",[]).append(dict(utc=datetime.now(timezone.utc).isoformat(),gpu=nvidia_smi()))
        meta["phase_status"]["C"]="resumed_running"
        manifest_path.write_text(json.dumps(meta,indent=2,default=str))
    else:
        config_path.write_text(json.dumps(config, indent=2))
        meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
            phase_status={"A": "historical evidence audit", "B": "multi-projection prototype",
                          "C": "running", "D": "not_run_pending_phase_C_gate",
                          "E": "not_run_pending_phase_C_gate"},
            branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
            source_hashes=source_hashes(), gpu_before=nvidia_smi(), gpu_runtime=gpu,
            python=sys.version, platform=platform.platform(), packages=package_versions(),
            protocol=protocol, failures=[], started_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(meta, indent=2, default=str))
    historical_gate = ROOT / "outputs/20261027-adaptive-four-phase-study/phase4_ann_and_combined/real_validation_pilot_device_refresh_v2/gate_summary.csv"
    if historical_gate.exists() and not args.resume:
        import shutil
        shutil.copyfile(historical_gate, output / "phaseA_ann_diagnostics/historical_projection_gate_summary.csv")
        prior_rows=list(csv.DictReader(historical_gate.open()))
        (output / "phaseA_ann_diagnostics/diagnostic.md").write_text(
            "# Existing GPU projection ANN evidence\n\n"
            f"Source: `{historical_gate.relative_to(ROOT)}` (copied byte-for-byte for this study).\n\n"
            f"The archived gate contains {len(prior_rows)} snapshot rows. "
            "Its single random projection narrows candidates by 1-D projected distance, then recomputes selected squared distances exactly on the GPU. "
            "Projection distance is not a lower bound on Euclidean distance, so poor recall is an algorithmic limitation of the shortlist; it is not evidence of inaccurate selected-distance arithmetic. "
            "The old index is sorted/built on CPU and refreshed by copying moving centroids to host, rebuilding, and uploading. "
            "The old gate's row-level results are retained unchanged; see the copied CSV.\n")
    elif not args.resume:
        (output / "phaseA_ann_diagnostics/diagnostic.md").write_text(
            "Historical gate summary not found at the expected path; historical evidence audit incomplete.\n")
    if not args.resume:
      (output / "phaseB_new_ann_backends/design.md").write_text(
        "# Bounded ANN alternative\n\n"
        "This study prototypes one alternative: four independently seeded random projections. For each point, the backend finds the nearest `effort/4` projection keys per direction, canonicalizes duplicate centroid rows, deduplicates candidate IDs, then evaluates exact float64 squared distances on the GPU. Effort is an upper bound on candidate distance evaluations, not an HNSW-style graph parameter. If effort reaches K, the backend explicitly labels the route as exact fallback. The projection directions and sorted lists are built on CPU and refreshed after centroid movement; index construction/refresh are included in measured lifecycle time.\n\n"
        "No second new strategy was implemented: the existing benchmark evidence makes the key uncertainty multi-view recall versus index/query overhead. A second backend before resolving that uncertainty would expand scope without a distinct justified hypothesis.\n")
    raw_path = output / "phaseC_assignment_benchmarks/individual_runs.csv"
    raw_fields=["scene","n","d","k","seed","repetition","snapshot","method","effort","search_mode","X_sha256","centroids_sha256","external_complete_s","query_s","index_build_s","index_refresh_s","transfer_in_s","transfer_out_s","strict_label_agreement","tie_aware_distance_agreement","mean_excess_squared_distance","max_excess_squared_distance","assignment_objective_gap","candidate_distance_evaluations","exhaustive_distance_evaluations","candidate_fraction","index_bytes","quality_status"]
    if args.resume:
        with raw_path.open(newline="") as f: rows=list(csv.DictReader(f))
        numeric={"n":int,"d":int,"k":int,"seed":int,"repetition":int,"external_complete_s":float,"query_s":float,
            "index_build_s":float,"index_refresh_s":float,"strict_label_agreement":float,"tie_aware_distance_agreement":float,
            "mean_excess_squared_distance":float,"max_excess_squared_distance":float,"assignment_objective_gap":float,
            "candidate_distance_evaluations":int,"exhaustive_distance_evaluations":int,"candidate_fraction":float,"index_bytes":int}
        for row in rows:
            row["effort"]=None if row["effort"]=="" else int(row["effort"])
            for field,cast in numeric.items(): row[field]=cast(row[field])
            for field in ("transfer_in_s","transfer_out_s"):
                row[field]=None if row[field] in ("","None") else float(row[field])
        with (output / "phaseC_assignment_benchmarks/failures.csv").open(newline="") as f:
            failures=list(csv.DictReader(f))
    else:
        rows, failures = [], []
        write_csv(raw_path, [], fields=raw_fields)
        write_csv(output / "phaseC_assignment_benchmarks/failures.csv", [], fields=["scene","k","seed","repeat","method","effort","phase","error"])
    complete_keys=set()
    for row in rows:
        complete_keys.add((row["scene"],int(row["k"]),int(row["seed"]),int(row["repetition"]),row["method"],row["effort"]))
    start_all = perf_counter()
    try:
        configs_by_k = {}
        for k in K_VALUES:
            configs = [("exact_cuda", None)]
            configs += [("single_projection", e) for e in OLD_EFFORTS if e < k]
            configs += [("multi_projection", e) for e in MULTI_EFFORTS if e < k]
            configs_by_k[k] = configs
        total_cases = sum(len(configs_by_k[k]) for k in K_VALUES) * len(scenes) * len(seeds) * args.repeats
        case_counter = len(complete_keys)
        for scene in scenes:
            X, scene_meta = load_scene(scene)
            X_hash = sha_array(X)
            (output / "phaseC_assignment_benchmarks" / f"{scene}_scene_metadata.json").write_text(json.dumps(scene_meta, indent=2))
            for k in K_VALUES:
                for seed in seeds:
                    C0 = initialize_centroids(X, k, seed=seed)
                    # One real full-data CUDA Lloyd update creates a genuinely
                    # moved centroid snapshot; this setup is outside assignment timings.
                    updated = fit_cuda(X, C0, FitConfig(k=k, seed=seed, max_steps=1, tolerance=0.0))
                    C1 = np.ascontiguousarray(updated.centroids, dtype=np.float32)
                    if np.array_equal(C0, C1):
                        state_note = "one CUDA Lloyd update produced identical centroids"
                    else:
                        state_note = "one full-data exact CUDA Lloyd update"
                    snapshots = {"initial": C0, "after_one_exact_cuda_lloyd_update": C1}
                    ref_by_state = {}
                    for snap, C in snapshots.items():
                        ref_cpu = assign_exact(X, C)
                        ref_gpu = _exact_query(X, C)
                        np.testing.assert_array_equal(ref_cpu.labels, ref_gpu["labels"])
                        np.testing.assert_allclose(ref_cpu.squared_distances, ref_gpu["distances"], rtol=1e-12, atol=1e-12)
                        ref_by_state[snap] = ref_gpu
                    configs = configs_by_k[k]
                    _warm(X, C0, C1, configs, seed, repeats=args.warmups)
                    for rep in range(args.repeats):
                        order_rng = np.random.default_rng(np.random.SeedSequence([72631, K_VALUES.index(k), seed, rep, scenes.index(scene)]))
                        ordered = list(configs)
                        order_rng.shuffle(ordered)
                        for method, effort in ordered:
                            case_key=(scene,k,seed,rep,method,effort)
                            if case_key in complete_keys:
                                continue
                            if perf_counter()-start_all >= args.budget_seconds:
                                raise TimeoutError(f"study budget exhausted after {case_counter}/{total_cases} scene/K/seed/repetition cases")
                            try:
                                if method == "exact_cuda":
                                    # C1 is the centroid state produced on-device
                                    # by the preceding Lloyd update in an iterative
                                    # fit. Upload it before the timing interval for
                                    # both exact and indexed methods; ANN still pays
                                    # its D2H/CPU rebuild/index upload on refresh.
                                    resident_C1 = cuda.to_device(C1)
                                    cuda.synchronize()
                                    outputs = [_exact_query(X, C0), _exact_query(X, C1, resident_C1)]
                                else:
                                    cls = ProjectionCUDAIndex if method == "single_projection" else MultiProjectionCUDAIndex
                                    outputs = _ann_lifecycle(cls, X, C0, C1, effort, seed)
                                appended=[]
                                for snap, C, result in zip(snapshots, snapshots.values(), outputs):
                                    row = _quality_row(result["labels"], result["distances"],
                                        ref_by_state[snap]["labels"], ref_by_state[snap]["distances"],
                                        X, C, method, effort, scene, k, seed, rep, snap,
                                        X_hash, sha_array(C), len(X), result["candidate_evaluations"],
                                        result["external_s"], result["query_s"], result["index_build_s"],
                                        result["index_refresh_s"], result.get("transfer_in_s"),
                                        result.get("transfer_out_s"), result["index_bytes"])
                                    rows.append(row);appended.append(row)
                                with raw_path.open("a",newline="") as stream:
                                    writer=csv.DictWriter(stream,fieldnames=raw_fields);writer.writerows(appended)
                                complete_keys.add(case_key)
                                case_counter += 1
                            except Exception as exc:
                                failures.append(dict(scene=scene,k=k,seed=seed,repeat=rep,
                                    method=method,effort=effort,phase="measurement",
                                    error=f"{type(exc).__name__}: {exc}"))
                                write_csv(output / "phaseC_assignment_benchmarks/failures.csv", failures)
                                raise
        # Summaries use repeated measurement medians; paired lifecycle speedup
        # sums the two state-specific external costs for the same seed/repeat.
        summary=[]
        keys=sorted({(r["scene"],r["k"],r["seed"],r["method"],r["effort"]) for r in rows}, key=str)
        for scene,k,seed,method,effort in keys:
            ss=[r for r in rows if (r["scene"],r["k"],r["seed"],r["method"],r["effort"])==(scene,k,seed,method,effort)]
            by_rep={r["repetition"]:r for r in ss if r["snapshot"]=="initial"}
            moved={r["repetition"]:r for r in ss if r["snapshot"]=="after_one_exact_cuda_lloyd_update"}
            times=[by_rep[i]["external_complete_s"]+moved[i]["external_complete_s"] for i in by_rep]
            summary.append(dict(scene=scene,k=k,seed=seed,method=method,effort=effort,
                repetitions=len(times),median_two_state_lifecycle_s=statistics.median(times),
                q1_two_state_lifecycle_s=float(np.percentile(times,25)),q3_two_state_lifecycle_s=float(np.percentile(times,75)),
                median_initial_agreement=statistics.median(r["strict_label_agreement"] for r in ss if r["snapshot"]=="initial"),
                median_moved_agreement=statistics.median(r["strict_label_agreement"] for r in ss if r["snapshot"]!="initial"),
                median_initial_objective_gap=statistics.median(r["assignment_objective_gap"] for r in ss if r["snapshot"]=="initial"),
                median_moved_objective_gap=statistics.median(r["assignment_objective_gap"] for r in ss if r["snapshot"]!="initial"),
                median_candidate_fraction=statistics.median(r["candidate_fraction"] for r in ss)))
        write_csv(output / "phaseC_assignment_benchmarks/method_summary.csv", summary)
        gate_rows=[]
        for scene,k in sorted({(r["scene"],r["k"]) for r in rows}):
            exact={ (r["seed"],r["repetition"],r["snapshot"]):r for r in rows if r["scene"]==scene and r["k"]==k and r["method"]=="exact_cuda" }
            for method in ("single_projection","multi_projection"):
                efforts=sorted({r["effort"] for r in rows if r["scene"]==scene and r["k"]==k and r["method"]==method})
                for effort in efforts:
                    ann=[r for r in rows if r["scene"]==scene and r["k"]==k and r["method"]==method and r["effort"]==effort]
                    lifecycle=[]
                    for seed in seeds:
                        for rep in range(args.repeats):
                            a=next(r for r in ann if r["seed"]==seed and r["repetition"]==rep and r["snapshot"]=="initial")
                            b=next(r for r in ann if r["seed"]==seed and r["repetition"]==rep and r["snapshot"]!="initial")
                            e1=exact[(seed,rep,"initial")]
                            e2=exact[(seed,rep,"after_one_exact_cuda_lloyd_update")]
                            lifecycle.append((e1["external_complete_s"]+e2["external_complete_s"])/(a["external_complete_s"]+b["external_complete_s"]))
                    recalls=[r["strict_label_agreement"] for r in ann]
                    gaps=[r["assignment_objective_gap"] for r in ann]
                    per_seed={seed:statistics.median(lifecycle[i*args.repeats:(i+1)*args.repeats])
                              for i,seed in enumerate(seeds)}
                    rng=np.random.default_rng(np.random.SeedSequence([8441,k,seeds[0],effort]))
                    seed_vals=np.asarray(list(per_seed.values()),dtype=np.float64)
                    boot=np.median(seed_vals[rng.integers(0,len(seed_vals),(10000,len(seed_vals)))],axis=1)
                    ci_low=float(np.quantile(boot,.025))
                    pass_quality=(min(recalls)>=GATE["strict_agreement_min"] and max(gaps)<=GATE["assignment_objective_gap_max"])
                    speed=statistics.median(lifecycle)
                    gate_rows.append(dict(scene=scene,k=k,method=method,effort=effort,
                        centroid_states=2,n_seed_repetitions=len(ann),min_strict_agreement=min(recalls),
                        max_objective_gap=max(gaps),median_two_state_speedup=speed,
                        median_seed_paired_speedup=statistics.median(per_seed.values()),
                        seed_bootstrap_speedup_ci95_low=ci_low,
                        passes_quality=pass_quality,passes_speed=speed>=GATE["median_paired_lifecycle_speedup_min"],
                        passes_gate=pass_quality and speed>=GATE["median_paired_lifecycle_speedup_min"] and
                                    ci_low>GATE["seed_bootstrap_speedup_ci95_lower_gt"]))
        write_csv(output / "phaseC_assignment_benchmarks/gate_summary.csv", gate_rows)
        passing_dev={(r["k"],r["method"],r["effort"]) for r in gate_rows if r["scene"]=="31UFS" and r["passes_gate"]}
        validation_pass={(r["k"],r["method"],r["effort"]) for r in gate_rows if r["scene"]=="31UFU" and r["passes_gate"]}
        proceed=bool(passing_dev & validation_pass)
        meta["phase_status"]["C"]="complete"
        meta["phase_status"]["D"]=("eligible_for_bounded_pilot" if proceed else "skipped_gate_failed")
        meta["phase_status"]["E"]=("eligible_only_if_D_ann_backend_validated" if proceed else "skipped_gate_failed")
        meta["gate_result"]={"dev_passing_configs":sorted(map(str,passing_dev)),
            "validation_passing_configs":sorted(map(str,validation_pass)),
            "same_config_passed_both_scenes":sorted(map(str,passing_dev & validation_pass)),
            "decision":"eligible for separately approved/ bounded end-to-end ANN pilot" if proceed else
                       "STOP: no configuration passed quality and maintenance-inclusive runtime gates on both scenes"}
        meta["complete"]=True
    except Exception as exc:
        meta["failures"].append(f"{type(exc).__name__}: {exc}")
        write_csv(output / "phaseC_assignment_benchmarks/failures.csv", failures)
        raise
    finally:
        meta["elapsed_wall_s"]=perf_counter()-start_all
        meta["gpu_after"]=nvidia_smi()
        meta["finished_utc"]=datetime.now(timezone.utc).isoformat()
        (output / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))
        for phase in ("phaseA_ann_diagnostics","phaseB_new_ann_backends"):
            (output / phase / "README.md").write_text(
                "See final_summary/final_summary.md for status and interpretation.\n")
        (output / "phaseD_end_to_end_ann/STATUS.md").write_text("Not executed unless the phase C gate passes.\n")
        (output / "phaseE_combined_adaptation/STATUS.md").write_text("Not executed unless phase D establishes a viable ANN backend.\n")
        checksum_lines=[]
        for p in sorted(output.rglob("*")):
            if p.is_file() and p.name!="checksums.sha256":
                checksum_lines.append(f"{sha_bytes(p.read_bytes())}  {p.relative_to(output)}\n")
        (output/"checksums.sha256").write_text("".join(checksum_lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/20261027-ann-optimization-study")
    parser.add_argument("--scenes",nargs="+",choices=tuple(SCENES),default=["31UFS","31UFU"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmups",type=int,default=1)
    parser.add_argument("--budget-seconds",type=float,default=900)
    parser.add_argument("--pilot",action="store_true",help="label this bounded three-seed feasibility run as a pilot")
    parser.add_argument("--resume",action="store_true",help="resume an incomplete archive with the same frozen protocol; appends only missing paired cases")
    args=parser.parse_args()
    if args.repeats<2 or args.warmups<1 or not args.seeds or any(s<0 for s in args.seeds):
        parser.error("need nonnegative seeds, at least 2 measured repetitions, and >=1 warmup")
    run(args)


if __name__=="__main__":
    main()
