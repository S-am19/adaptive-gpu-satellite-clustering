"""Optional CUDA sorted random-projection ANN, with explicit exact small-K mode.

Index construction/sorting runs on CPU over K centroids; queries run on CUDA.
Effort is the number of candidate centroids evaluated, NOT HNSW efSearch.
One projection can have poor recall: no recall or performance benefit is assumed.
"""
from time import perf_counter

import numpy as np
from numba import cuda, float64

from .contracts import AssignmentResult, validate_integer, validate_points_centroids
from .minibatch_cuda import _assign_indexed_kernel, _blocks, _THREADS


@cuda.jit
def _projected_query(points, indices, centroids, direction, keys, ids, canonical,
                     effort, labels, distances):
    for i in range(cuda.grid(1), len(indices), cuda.gridsize(1)):
        point = indices[i]
        projection = 0.0
        for d in range(points.shape[1]):
            projection += float64(points[point, d]) * direction[d]
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < projection:
                lo = mid + 1
            else:
                hi = mid
        left, right = lo - 1, lo
        best, best_id = float('inf'), 2147483647
        for _ in range(effort):
            take_left = False
            if right >= len(keys):
                take_left = True
            elif left >= 0:
                dl, dr = abs(projection - keys[left]), abs(projection - keys[right])
                take_left = dl < dr or (dl == dr and ids[left] < ids[right])
            if take_left:
                j = ids[left]
                left -= 1
            else:
                j = ids[right]
                right += 1
            j = canonical[j]
            distance = 0.0
            for d in range(points.shape[1]):
                delta = float64(points[point, d]) - float64(centroids[j, d])
                distance += delta * delta
            if distance < best or (distance == best and j < best_id):
                best, best_id = distance, j
        labels[i], distances[i] = best_id, best


class ProjectionCUDAIndex:
    """AssignmentBackend host adapter plus versioned resident-query API.

    Host assign verifies centroid contents. Resident callers must increment their
    centroid version after EVERY mutation and refresh before querying. Query uses
    the index's owned centroid snapshot; a mismatched version fails explicitly.
    Ledger durations overlap public AssignmentResult accounting; never add both.
    """
    def __init__(self, *, seed=0, ann_effort=16, exact_threshold=8, query_batch_size=4096):
        for name, value, minimum in [('seed', seed, 0), ('ann_effort', ann_effort, 1),
                                      ('exact_threshold', exact_threshold, 0), ('query_batch_size', query_batch_size, 1)]:
            validate_integer(name, value, minimum)
        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError('ProjectionCUDAIndex requires real CUDA hardware')
        self.seed, self.ann_effort = int(seed), int(ann_effort)
        self.exact_threshold, self.query_batch_size = int(exact_threshold), int(query_batch_size)
        self._centroids = None
        self.version = 0
        self._pending = 0.0
        self._times = dict(initial_build_s=0., rebuild_s=0., query_s=0.,
                           initial_build_count=0, rebuild_count=0, query_count=0,
                           centroid_distance_evaluations=0, projection_dot_products=0)

    @property
    def timings(self):
        return dict(self._times, pending_index_update_s=self._pending)

    def effective_effort(self, effort=None):
        value = self.ann_effort if effort is None else effort
        validate_integer('effort', value)
        if self._centroids is None:
            raise RuntimeError('prepare must precede query')
        k = len(self._centroids)
        return k if k <= self.exact_threshold else min(int(value), k)

    def search_mode(self, effort=None):
        budget = self.effective_effort(effort)
        if len(self._centroids) <= self.exact_threshold:
            return 'exact_small_k'
        return 'exact_exhaustive_budget' if budget == len(self._centroids) else 'approximate_projection_window'

    def prepare(self, centroids, effort=None):
        validate_points_centroids(centroids, centroids)
        if effort is not None:
            validate_integer('effort', effort)
        start = perf_counter()
        snapshot = centroids.copy()
        direction = np.random.default_rng(self.seed).normal(size=snapshot.shape[1])
        direction /= np.linalg.norm(direction)
        # Explicit sum matches the dimensional reduction order of the GPU query.
        keys = np.zeros(len(snapshot), np.float64)
        for d in range(snapshot.shape[1]):
            keys += snapshot[:, d].astype(np.float64) * direction[d]
        ids = np.lexsort((np.arange(len(snapshot)), keys)).astype(np.int32)
        _, first, inverse = np.unique(snapshot, axis=0, return_index=True, return_inverse=True)
        canonical = np.ascontiguousarray(first[inverse], dtype=np.int32)
        arrays=(snapshot,direction,keys[ids],ids,canonical)
        required=sum(a.nbytes for a in arrays)
        free,_=cuda.current_context().get_memory_info()
        if required>.8*free: raise MemoryError('Projection index refresh exceeds available device memory')
        buffers = [cuda.to_device(a) for a in arrays]
        cuda.synchronize()
        prefix = 'initial_build' if self._centroids is None else 'rebuild'
        elapsed = perf_counter() - start
        self._times[prefix+'_s'] += elapsed
        self._times[prefix+'_count'] += 1
        self._pending += elapsed
        self.device_index_bytes=required
        self._centroids = snapshot
        self._C, self._direction, self._keys, self._ids, self._canonical = buffers
        if effort is not None:
            self.ann_effort = int(effort)

    def refresh(self, new_centroids):
        if self._centroids is None:
            raise RuntimeError('prepare must precede refresh')
        self.prepare(new_centroids)
        self.version += 1

    def refresh_device(self, centroids, version):
        validate_integer('version', version)
        if version <= self.version:
            raise ValueError('Centroid version must advance after each update')
        start = perf_counter()
        host = centroids.copy_to_host()
        cuda.synchronize()
        transfer = perf_counter()-start
        self.refresh(host)
        self.version = int(version)
        self._times['rebuild_s'] += transfer
        self._pending += transfer

    def assign_device(self, points, indices, labels, distances, *, version, effort=None):
        if version != self.version:
            raise ValueError('stale index: refresh to the current centroid version')
        budget = self.effective_effort(effort)
        start = perf_counter()
        if self.search_mode(effort) != 'approximate_projection_window':
            _assign_indexed_kernel[_blocks(len(indices)), _THREADS](points, indices, self._C, labels, distances)
        else:
            _projected_query[_blocks(len(indices)), _THREADS](points, indices, self._C,
                self._direction, self._keys, self._ids, self._canonical, budget, labels, distances)
        cuda.synchronize()
        self._times['query_s'] += perf_counter()-start
        self._times['query_count'] += 1
        self._times['centroid_distance_evaluations'] += int(len(indices))*budget
        if self.search_mode(effort)=='approximate_projection_window':
            self._times['projection_dot_products'] += int(len(indices))
        pending, self._pending = self._pending, 0.0
        return AssignmentResult(labels, distances, perf_counter()-start, pending)

    def assign(self, points, centroids, effort=None):
        validate_points_centroids(points, centroids)
        if self._centroids is None:
            raise RuntimeError('prepare must precede assign')
        if not np.array_equal(centroids, self._centroids):
            raise ValueError('stale index: call refresh')
        start = perf_counter()
        labels = np.empty(len(points), np.int32)
        distances = np.empty(len(points), np.float64)
        pending = 0.
        for begin in range(0, len(points), self.query_batch_size):
            end = min(begin+self.query_batch_size, len(points))
            data = cuda.to_device(points[begin:end])
            indices = cuda.to_device(np.arange(end-begin, dtype=np.int32))
            out_labels = cuda.device_array(end-begin, np.int32)
            out_distances = cuda.device_array(end-begin, np.float64)
            result = self.assign_device(data, indices, out_labels, out_distances, version=self.version, effort=effort)
            labels[begin:end], distances[begin:end] = out_labels.copy_to_host(), out_distances.copy_to_host()
            pending += result.index_update_time_s
        cuda.synchronize()
        return AssignmentResult(labels, distances, perf_counter()-start, pending)
