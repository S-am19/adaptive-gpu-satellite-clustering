"""Experimental multi-projection CUDA candidate assignment.

This isolated alternative uses several CPU-built sorted random projections to
form a deduplicated shortlist. Candidate distances are recomputed exactly on
the GPU. It is approximate unless the configured candidate budget reaches K,
in which case the backend explicitly dispatches to exact assignment.
"""
from time import perf_counter

import numpy as np
from numba import cuda, float64, int32

from .contracts import AssignmentResult, validate_integer, validate_points_centroids
from .minibatch_cuda import _assign_indexed_kernel, _blocks, _THREADS

_PROJECTIONS = 4
_MAX_CANDIDATES = 32


@cuda.jit
def _fill_candidate_counts(counts, value):
    for i in range(cuda.grid(1), counts.size, cuda.gridsize(1)):
        counts[i] = value


@cuda.jit
def _multi_projection_query(points, indices, centroids, directions, keys, ids,
                            canonical, per_projection, labels, distances,
                            candidate_counts):
    for i in range(cuda.grid(1), indices.shape[0], cuda.gridsize(1)):
        point = indices[i]
        candidate_ids = cuda.local.array(_MAX_CANDIDATES, dtype=int32)
        candidate_count = 0
        for p in range(directions.shape[0]):
            projection = 0.0
            for d in range(points.shape[1]):
                projection += float64(points[point, d]) * directions[p, d]
            lo, hi = 0, keys.shape[1]
            while lo < hi:
                mid = (lo + hi) // 2
                if keys[p, mid] < projection:
                    lo = mid + 1
                else:
                    hi = mid
            left, right = lo - 1, lo
            for _ in range(per_projection):
                take_left = False
                if right >= keys.shape[1]:
                    take_left = True
                elif left >= 0:
                    dl = abs(projection - keys[p, left])
                    dr = abs(projection - keys[p, right])
                    take_left = dl < dr or (dl == dr and ids[p, left] < ids[p, right])
                if take_left:
                    candidate = canonical[ids[p, left]]
                    left -= 1
                else:
                    candidate = canonical[ids[p, right]]
                    right += 1
                duplicate = False
                for q in range(candidate_count):
                    if candidate_ids[q] == candidate:
                        duplicate = True
                        break
                if not duplicate and candidate_count < _MAX_CANDIDATES:
                    candidate_ids[candidate_count] = candidate
                    candidate_count += 1

        best_distance, best_label = float("inf"), 2147483647
        for q in range(candidate_count):
            j = candidate_ids[q]
            distance = 0.0
            for d in range(points.shape[1]):
                delta = float64(points[point, d]) - float64(centroids[j, d])
                distance += delta * delta
            if distance < best_distance or (distance == best_distance and j < best_label):
                best_distance, best_label = distance, j
        labels[i], distances[i] = best_label, best_distance
        candidate_counts[i] = candidate_count


class MultiProjectionCUDAIndex:
    """Versioned resident-query adapter for a four-projection shortlist."""

    def __init__(self, *, seed=0, ann_effort=8, exact_threshold=8):
        validate_integer("seed", seed, 0)
        validate_integer("ann_effort", ann_effort, _PROJECTIONS)
        validate_integer("exact_threshold", exact_threshold, 0)
        if ann_effort % _PROJECTIONS:
            raise ValueError(f"ann_effort must be a multiple of {_PROJECTIONS}")
        if ann_effort > _MAX_CANDIDATES:
            raise ValueError(f"ann_effort must be <= {_MAX_CANDIDATES}")
        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError("MultiProjectionCUDAIndex requires real CUDA hardware")
        self.seed, self.ann_effort = int(seed), int(ann_effort)
        self.exact_threshold = int(exact_threshold)
        self.version = 0
        self._centroids = None
        self._pending = 0.0
        self._times = dict(initial_build_s=0.0, rebuild_s=0.0, query_s=0.0,
                           initial_build_count=0, rebuild_count=0, query_count=0)

    @property
    def timings(self):
        return dict(self._times, pending_index_update_s=self._pending)

    @property
    def index_bytes(self):
        return 0 if self._centroids is None else self._index_bytes

    def search_mode(self, effort=None):
        budget = self.ann_effort if effort is None else effort
        self._validate_effort(budget)
        k = 0 if self._centroids is None else len(self._centroids)
        if k <= self.exact_threshold:
            return "exact_small_k"
        if budget >= k:
            return "exact_budget_fallback"
        return "approximate_multi_projection"

    def prepare(self, centroids, effort=None):
        validate_points_centroids(centroids, centroids)
        if effort is not None:
            self._validate_effort(effort)
        started = perf_counter()
        snapshot = np.ascontiguousarray(centroids, dtype=np.float32).copy()
        directions = np.random.default_rng(self.seed).normal(
            size=(_PROJECTIONS, snapshot.shape[1])).astype(np.float64)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        projected = snapshot.astype(np.float64) @ directions.T
        keys = np.empty((_PROJECTIONS, len(snapshot)), dtype=np.float64)
        ids = np.empty((_PROJECTIONS, len(snapshot)), dtype=np.int32)
        for p in range(_PROJECTIONS):
            ids[p] = np.lexsort((np.arange(len(snapshot)), projected[:, p]))
            keys[p] = projected[ids[p], p]
        _, first, inverse = np.unique(snapshot, axis=0, return_index=True, return_inverse=True)
        canonical = np.ascontiguousarray(first[inverse], dtype=np.int32)
        arrays = (snapshot, directions, keys, ids, canonical)
        needed = sum(a.nbytes for a in arrays)
        free, _ = cuda.current_context().get_memory_info()
        if needed > 0.8 * free:
            raise MemoryError("Multi-projection index exceeds 80% of currently free device memory")
        device_arrays = [cuda.to_device(a) for a in arrays]
        cuda.synchronize()
        category = "initial_build" if self._centroids is None else "rebuild"
        elapsed = perf_counter() - started
        self._times[category + "_s"] += elapsed
        self._times[category + "_count"] += 1
        self._pending += elapsed
        self._centroids = snapshot
        self._C, self._directions, self._keys, self._ids, self._canonical = device_arrays
        self._index_bytes = needed
        if effort is not None:
            self.ann_effort = int(effort)

    def _validate_effort(self, effort):
        validate_integer("effort", effort, _PROJECTIONS)
        if effort > _MAX_CANDIDATES or effort % _PROJECTIONS:
            raise ValueError(f"effort must be a multiple of {_PROJECTIONS} in 4..{_MAX_CANDIDATES}")

    def refresh(self, new_centroids):
        if self._centroids is None:
            raise RuntimeError("prepare must precede refresh")
        self.prepare(new_centroids)
        self.version += 1

    def refresh_device(self, centroids, version):
        validate_integer("version", version, 0)
        if version <= self.version:
            raise ValueError("centroid version must advance after every update")
        started = perf_counter()
        host = centroids.copy_to_host()
        cuda.synchronize()
        transfer = perf_counter() - started
        self.refresh(host)
        self.version = int(version)
        self._times["rebuild_s"] += transfer
        self._pending += transfer

    def assign_device(self, points, indices, labels, distances, candidate_counts,
                      *, version, effort=None):
        if self._centroids is None:
            raise RuntimeError("prepare must precede query")
        if int(version) != self.version:
            raise ValueError("stale index: refresh to current centroids")
        selected_effort = self.ann_effort if effort is None else effort
        self._validate_effort(selected_effort)
        selected_effort = int(selected_effort)
        started = perf_counter()
        if self.search_mode(selected_effort) != "approximate_multi_projection":
            _assign_indexed_kernel[_blocks(indices.size), _THREADS](
                points, indices, self._C, labels, distances)
            _fill_candidate_counts[_blocks(indices.size), _THREADS](
                candidate_counts, len(self._centroids))
        else:
            _multi_projection_query[_blocks(indices.size), _THREADS](
                points, indices, self._C, self._directions, self._keys,
                self._ids, self._canonical, selected_effort // _PROJECTIONS,
                labels, distances, candidate_counts)
        cuda.synchronize()
        elapsed = perf_counter() - started
        self._times["query_s"] += elapsed
        self._times["query_count"] += 1
        pending, self._pending = self._pending, 0.0
        return AssignmentResult(labels, distances, elapsed, pending)

    def assign(self, points, centroids, effort=None):
        validate_points_centroids(points, centroids)
        if self._centroids is None:
            raise RuntimeError("prepare must precede assign")
        if not np.array_equal(centroids, self._centroids):
            raise ValueError("stale index: call refresh")
        self._validate_effort(self.ann_effort if effort is None else effort)
        wall_started = perf_counter()
        labels = np.empty(len(points), dtype=np.int32)
        distances = np.empty(len(points), dtype=np.float64)
        candidate_counts = np.empty(len(points), dtype=np.int32)
        pending = 0.0
        for begin in range(0, len(points), 4096):
            end = min(begin + 4096, len(points))
            device_points = cuda.to_device(np.ascontiguousarray(points[begin:end]))
            indices = cuda.to_device(np.arange(end - begin, dtype=np.int32))
            out_labels = cuda.device_array(end - begin, dtype=np.int32)
            out_distances = cuda.device_array(end - begin, dtype=np.float64)
            out_counts = cuda.device_array(end - begin, dtype=np.int32)
            result = self.assign_device(device_points, indices, out_labels,
                out_distances, out_counts, version=self.version, effort=effort)
            labels[begin:end] = out_labels.copy_to_host()
            distances[begin:end] = out_distances.copy_to_host()
            candidate_counts[begin:end] = out_counts.copy_to_host()
            pending += result.index_update_time_s
        cuda.synchronize()
        self.last_candidate_distance_evaluations = int(candidate_counts.sum())
        return AssignmentResult(labels, distances, perf_counter() - wall_started, pending)
