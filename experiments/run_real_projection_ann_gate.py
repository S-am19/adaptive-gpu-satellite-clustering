"""Bounded real-scene crossover gate for CUDA projection assignment ANN.

This measures assignment only. Each row's end-to-end cost includes point/index
transfers, index construction or refresh, query and label/distance downloads.
CUDA-event/index ledger fields are reported separately and never summed into
the external wall time.
"""
import argparse
import csv
import hashlib
import json
import sys
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numba import cuda
from pcap.assignments_ann_cuda import ProjectionCUDAIndex
from pcap.assignments_exact import assign_exact
from pcap.initialization import initialize_centroids
from pcap.minibatch_cuda import _assign_indexed_kernel, _blocks, _THREADS
from run_suite import provenance, snapshot_sources


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _moved_snapshot(X, C, seed, batch=16384):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 901]))
    idx = rng.choice(len(X), size=min(batch, len(X)), replace=False)
    ref = assign_exact(X[idx], C)
    sums = np.zeros(C.shape, np.float64)
    counts = np.bincount(ref.labels, minlength=len(C)).astype(np.int64)
    np.add.at(sums, ref.labels, X[idx].astype(np.float64))
    moved = C.copy()
    present = counts > 0
    moved[present] = (sums[present] / counts[present, None]).astype(np.float32)
    return np.ascontiguousarray(moved)


def _exact_cuda(X, C):
    n = len(X)
    wall = perf_counter()
    dx = cuda.to_device(X)
    dc = cuda.to_device(C)
    indices = cuda.to_device(np.arange(n, dtype=np.int32))
    labels = cuda.device_array(n, np.int32)
    distances = cuda.device_array(n, np.float64)
    cuda.synchronize()
    _assign_indexed_kernel[_blocks(n), _THREADS](dx, indices, dc, labels, distances)
    cuda.synchronize()
    host_labels, host_distances = labels.copy_to_host(), distances.copy_to_host()
    cuda.synchronize()
    return host_labels, host_distances, perf_counter() - wall


def _ann_cuda(X, C, effort, seed, prior_centroids=None,
              updated_device_centroids=None):
    wall = perf_counter()
    index = ProjectionCUDAIndex(seed=seed, ann_effort=effort)
    if prior_centroids is None:
        index.prepare(C, effort)
    else:
        index.prepare(prior_centroids, effort)
        if updated_device_centroids is None:
            raise ValueError("updated device centroids are required to measure refresh")
        index.refresh_device(updated_device_centroids, version=index.version + 1)
    n = len(X)
    dx = cuda.to_device(X)
    indices = cuda.to_device(np.arange(n, dtype=np.int32))
    labels = cuda.device_array(n, np.int32)
    distances = cuda.device_array(n, np.float64)
    cuda.synchronize()
    query_start = perf_counter()
    index.assign_device(dx, indices, labels, distances,
                        version=index.version, effort=effort)
    query_s = perf_counter() - query_start
    host_labels, host_distances = labels.copy_to_host(), distances.copy_to_host()
    cuda.synchronize()
    total = perf_counter() - wall
    ledger = index.timings
    return host_labels, host_distances, total, query_s, ledger


def run(args):
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("real CUDA GPU required; simulator/fallback is disabled")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    scene_dir = args.scene_dir.resolve()
    X = np.load(scene_dir / "X.npy", mmap_mode="r")
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 4 or not np.isfinite(X).all():
        raise ValueError("expected finite Sentinel-2 validation features [N,4]")
    meta = json.loads((scene_dir / "metadata.json").read_text())
    X_hash = sha(X.tobytes())
    manifest = dict(provenance(), complete=False, failures=[],
        stage="real validation scene assignment-only ANN gate",
        protocol={"scene": meta.get("scene_id"), "n": len(X), "d": X.shape[1],
            "k_values": args.clusters, "seeds": args.seeds,
            "efforts": "{1,4,8,16,32} filtered to effort<K",
            "snapshots": "seeded init and one exact CPU mini-batch centroid update",
            "warmups": args.warmups, "repetitions": args.repeats,
            "gate_predeclared": {"strict_recall_min": 0.95,
                "mean_objective_gap_max": 0.01,
                "median_maintenance_inclusive_speedup_min": 1.0},
            "target_is_assignment_quality_only": True,
            "X_sha256": X_hash,
            "timing": "external wall includes data/index transfers, build/refresh, query, output copy; query is a subset"})
    (out / "config.json").write_text(json.dumps(manifest["protocol"], indent=2, default=str))
    snapshot_sources(out / "source", manifest["source_hashes"])
    raw, failures = [], []
    start = perf_counter()
    try:
        # Warm each kernel family before measurements; no warm-up is counted.
        warm_X = X[:min(len(X), 4096)]
        for k in args.clusters:
            for seed in args.seeds[:1]:
                C = initialize_centroids(X, k, seed=seed)
                _exact_cuda(warm_X, C)
                for effort in sorted({e for e in (1, 4, 8, 16, 32) if e < k}):
                    _ann_cuda(warm_X, C, effort, seed)
        for k in args.clusters:
            if k > len(X):
                failures.append({"k": k, "reason": "K exceeds N"})
                continue
            for seed in args.seeds:
                C0 = initialize_centroids(X, k, seed=seed)
                snapshots = [("initial", C0), ("after_one_minibatch_update", _moved_snapshot(X, C0, seed))]
                for snapshot_name, C in snapshots:
                    exact_host = assign_exact(X, C)
                    exact_labels, exact_distances, exact_wall = _exact_cuda(X, C)
                    np.testing.assert_array_equal(exact_labels, exact_host.labels)
                    np.testing.assert_allclose(exact_distances, exact_host.squared_distances,
                                               rtol=1e-12, atol=1e-12)
                    methods = [("exact", None)] + [("projection", e)
                        for e in sorted({e for e in (1, 4, 8, 16, 32) if e < k})]
                    for phase, count in (("warmup", args.warmups), ("measured", args.repeats)):
                        for rep in range(count):
                            # Balanced deterministic rotation by seed/snapshot/repetition.
                            offset = (seed + rep + (snapshot_name != "initial")) % len(methods)
                            order = methods[offset:] + methods[:offset]
                            for method, effort in order:
                                if phase == "warmup":
                                    if method == "exact": _exact_cuda(X, C)
                                    else: _ann_cuda(X, C, effort, seed)
                                    continue
                                prior = C0 if (method == "projection" and
                                               snapshot_name != "initial") else None
                                updated_device = cuda.to_device(C) if prior is not None else None
                                if updated_device is not None:
                                    cuda.synchronize()
                                t = perf_counter()
                                if method == "exact":
                                    labels, distances, ext = _exact_cuda(X, C)
                                    query = None
                                    ledger = {}
                                    mode = "exact"
                                    effective = k
                                else:
                                    labels, distances, ext, query, ledger = _ann_cuda(
                                        X, C, effort, seed, prior_centroids=prior,
                                        updated_device_centroids=updated_device)
                                    mode = "approximate_projection_window"
                                    effective = effort
                                # External wrapper is independently observed; ext is the same nested
                                # complete-call boundary and retained as an implementation check.
                                outer = perf_counter() - t
                                independent = np.sum((X.astype(np.float64) -
                                    C[labels].astype(np.float64)) ** 2, axis=1)
                                np.testing.assert_allclose(distances, independent, rtol=1e-12, atol=1e-12)
                                excess = np.maximum(distances - exact_host.squared_distances, 0.0)
                                raw.append(dict(k=k, seed=seed, snapshot=snapshot_name,
                                    snapshot_sha256=sha(C.tobytes()), X_sha256=X_hash,
                                    method=method, effort=effort, effective_candidates=effective,
                                    mode=mode, phase=phase, repeat=rep,
                                    external_complete_wall_s=outer, internal_complete_s=ext,
                                    query_s=query, index_build_s=ledger.get("initial_build_s"),
                                    index_rebuild_s=ledger.get("rebuild_s"),
                                    strict_label_agreement=float(np.mean(labels == exact_labels)),
                                    tie_aware_distance_agreement=float(np.mean(np.isclose(
                                        distances, exact_host.squared_distances, rtol=1e-10, atol=1e-12))),
                                    mean_excess_squared_distance=float(excess.mean()),
                                    max_excess_squared_distance=float(excess.max()),
                                    assignment_objective_gap=float((distances.sum()-exact_host.squared_distances.sum()) /
                                        max(float(exact_host.squared_distances.sum()), 1e-30)),
                                    candidate_distance_evaluations=len(X)*effective,
                                    exhaustive_distance_evaluations=len(X)*k,
                                    projection_dot_products=(len(X) if method == "projection" else 0),
                                    exact_cuda_wall_s=exact_wall if method != "exact" else outer))
        write_csv(out / "individual_runs.csv", raw)
        write_csv(out / "failures.csv", failures)
        (out / "checkpoints.csv").write_text(
            "checkpoint_kind,not_applicable_reason\n"
            "not_applicable,assignment-only experiment has one result per query and no iterative checkpoints\n")
        (out / "controller_events.csv").write_text(
            "event_type,not_applicable_reason\n"
            "not_applicable,no adaptive controller in the ANN feasibility gate\n")
        timing_rows = [dict(k=r["k"], seed=r["seed"], snapshot=r["snapshot"],
            method=r["method"], effort=r["effort"], phase=r["phase"], repeat=r["repeat"],
            external_complete_wall_s=r["external_complete_wall_s"],
            internal_complete_s=r["internal_complete_s"], query_s=r["query_s"],
            index_build_s=r["index_build_s"], index_rebuild_s=r["index_rebuild_s"],
            timing_scope="query is included in external complete wall; do not sum components")
            for r in raw]
        write_csv(out / "timing_breakdown.csv", timing_rows)
        summary_rows = []
        for key in sorted({(r["k"], r["snapshot"], r["method"], r["effort"])
                           for r in raw if r["phase"] == "measured"}, key=str):
            subset = [r for r in raw if (r["k"], r["snapshot"], r["method"], r["effort"]) == key
                      and r["phase"] == "measured"]
            row = dict(k=key[0], snapshot=key[1], method=key[2], effort=key[3], n=len(subset))
            for field in ("external_complete_wall_s", "strict_label_agreement",
                          "assignment_objective_gap", "mean_excess_squared_distance"):
                vals = [float(r[field]) for r in subset]
                row["median_" + field] = statistics.median(vals)
                row["q1_" + field] = float(np.percentile(vals, 25))
                row["q3_" + field] = float(np.percentile(vals, 75))
            summary_rows.append(row)
        write_csv(out / "method_summary.csv", summary_rows)
        gate_rows = []
        for k in sorted({int(r["k"]) for r in raw if r["phase"] == "measured"}):
            efforts = sorted({int(r["effort"]) for r in raw if r["method"] == "projection"
                              and int(r["k"]) == k and r["phase"] == "measured"})
            for effort in efforts:
                snapshot_pass = {}
                for snapshot in ("initial", "after_one_minibatch_update"):
                    candidates = [r for r in raw if r["method"] == "projection" and
                        r["phase"] == "measured" and int(r["k"]) == k and
                        int(r["effort"]) == effort and r["snapshot"] == snapshot]
                    exact = {int(r["seed"]): r for r in raw if r["method"] == "exact" and
                        r["phase"] == "measured" and int(r["k"]) == k and
                        r["snapshot"] == snapshot}
                    recall = statistics.median(float(r["strict_label_agreement"]) for r in candidates)
                    gap = statistics.median(float(r["assignment_objective_gap"]) for r in candidates)
                    speed = statistics.median(float(exact[int(r["seed"])]["external_complete_wall_s"]) /
                        float(r["external_complete_wall_s"]) for r in candidates)
                    passes = recall >= .95 and gap <= .01 and speed >= 1.0
                    snapshot_pass[snapshot] = passes
                    gate_rows.append(dict(k=k, effort=effort, snapshot=snapshot,
                        n_seeds=len(candidates), median_strict_agreement=recall,
                        median_objective_gap=gap, median_complete_speedup=speed,
                        passes_snapshot_gate=passes, passes_both_snapshot_gates=False))
                both = all(snapshot_pass.values())
                for row in gate_rows:
                    if row["k"] == k and row["effort"] == effort:
                        row["passes_both_snapshot_gates"] = both
        write_csv(out / "gate_summary.csv", gate_rows)
        passing = sorted({(r["k"], r["effort"]) for r in gate_rows
                          if r["passes_both_snapshot_gates"]})
        manifest["gate_decision"] = dict(
            criteria=dict(median_strict_agreement_min=.95,
                median_objective_gap_max=.01, median_complete_speedup_min=1.0,
                both_centroid_snapshots_required=True),
            passing_k_effort_pairs=passing,
            decision=("proceed to Stage B pilot" if passing else
                      "STOP: no K/effort passed the predeclared quality and cost gate"))
        manifest["complete"] = True
    except Exception as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")
        write_csv(out / "individual_runs.csv", raw)
        raise
    finally:
        manifest["measured_rows"] = len(raw)
        manifest["elapsed_wall_s"] = perf_counter() - start
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
        (out / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n"
            for p in sorted(files)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--clusters", type=int, nargs="+", default=[16,32,64,128,256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
