"""Optional CPU HNSW feasibility adapter; no CUDA or clustering driver.

Search uses float32 L2 internally. Returned distances use float64 arithmetic.
Only duplicate-centroid ties are canonicalized globally; general ANN ties need
not match exact lowest-index assignment. No distance-evaluation savings claimed.
"""

import importlib
from time import perf_counter

import numpy as np

from .contracts import (AssignmentResult, validate_features, validate_integer,
                        validate_points_centroids)


class HNSWCPUBackend:
    """AssignmentBackend adapter with explicit refresh and bounded query batches.

    ``ann_effort`` maps directly to HNSW ef (not a visit/distance-count limit).
    ``timings`` contains cumulative, disjoint build/rebuild/query durations.
    AssignmentResult.index_update_time_s reports pending maintenance exactly
    once, on the next successful assignment. It overlaps the cumulative ledger:
    use either accounting view, never sum the two. Unconsumed maintenance stays
    visible in timings even if no subsequent query occurs. Instances are not
    thread-safe; construction and queries explicitly use one CPU thread.
    """

    def __init__(self, *, seed=0, m=8, ef_construction=100, ann_effort=16,
                 query_batch_size=4096):
        for name, value, minimum in (("seed", seed, 0), ("m", m, 2),
                                     ("ef_construction", ef_construction, 1),
                                     ("ann_effort", ann_effort, 1),
                                     ("query_batch_size", query_batch_size, 1)):
            validate_integer(name, value, minimum)
        if seed > np.iinfo(np.int32).max:
            raise ValueError("seed must fit a signed int32 for HNSW construction")
        if ef_construction < m:
            raise ValueError("ef_construction must be >= m")
        self.seed, self.m, self.ef_construction = int(seed), int(m), int(ef_construction)
        self.ann_effort, self.query_batch_size = int(ann_effort), int(query_batch_size)
        self._index = self._centroids = self._canonical = None
        self._pending_update_s = 0.0
        self._timings = dict(initial_build_s=0.0, rebuild_s=0.0, query_s=0.0,
                             initial_build_count=0, rebuild_count=0, query_count=0)

    @property
    def timings(self):
        """A copy of the cumulative ledger; includes unconsumed maintenance."""
        return dict(self._timings, pending_index_update_s=self._pending_update_s)

    def _effort(self, effort):
        value = self.ann_effort if effort is None else effort
        validate_integer("effort", value)
        return int(value)

    @staticmethod
    def _validate_search_range(name, values):
        validate_features(name, values)
        # Bound every component delta by 2*limit, leaving ample float32 L2
        # headroom. M1/M2 support larger inputs; this optional backend rejects
        # them explicitly rather than silently overflowing its search metric.
        limit = np.sqrt(float(np.finfo(np.float32).max) / values.shape[1]) / 4
        if np.max(np.abs(values)) > limit:
            raise ValueError(f"{name} exceeds the supported float32 HNSW L2 range")

    def prepare(self, centroids, effort=None):
        """Build an owned snapshot; repeated prepare calls are counted rebuilds."""
        self._validate_search_range("centroids", centroids)
        validate_points_centroids(centroids, centroids)
        search_effort = self._effort(effort)
        try:
            hnswlib = importlib.import_module("hnswlib")
        except ImportError as exc:
            raise ImportError("HNSWCPUBackend requires optional hnswlib in the project .venv") from exc
        start = perf_counter()
        snapshot = centroids.copy()
        index = hnswlib.Index(space="l2", dim=snapshot.shape[1])
        index.init_index(max_elements=len(snapshot), M=self.m,
                         ef_construction=self.ef_construction, random_seed=self.seed)
        index.set_num_threads(1)
        index.add_items(snapshot, np.arange(len(snapshot)), num_threads=1)
        index.set_ef(search_effort)
        # Preserve all centroid IDs in the index; canonicalize exact duplicates.
        _, first, inverse = np.unique(snapshot, axis=0, return_index=True, return_inverse=True)
        canonical = np.ascontiguousarray(first[inverse], dtype=np.int32)
        elapsed = perf_counter() - start
        prefix = "initial_build" if self._index is None else "rebuild"
        self._timings[prefix + "_s"] += elapsed
        self._timings[prefix + "_count"] += 1
        self._pending_update_s += elapsed
        # Publish only a completely built index, so failed refresh is atomic.
        self._index, self._centroids, self._canonical = index, snapshot, canonical

    def refresh(self, new_centroids):
        if self._index is None:
            raise RuntimeError("prepare must precede refresh")
        self.prepare(new_centroids)

    def assign(self, points, centroids, effort=None):
        validate_points_centroids(points, centroids)
        search_effort = self._effort(effort)
        if self._index is None:
            raise RuntimeError("prepare must precede assign")
        if not np.array_equal(centroids, self._centroids):
            raise ValueError("stale index: call refresh with the current centroids")
        self._validate_search_range("points", points)
        start = perf_counter()
        self._index.set_ef(search_effort)
        labels = np.empty(len(points), dtype=np.int32)
        distances = np.empty(len(points), dtype=np.float64)
        for begin in range(0, len(points), self.query_batch_size):
            end = min(begin + self.query_batch_size, len(points))
            ids, _ = self._index.knn_query(points[begin:end], k=1, num_threads=1)
            ids = np.asarray(ids)
            if (ids.shape != (end - begin, 1) or not np.issubdtype(ids.dtype, np.integer)
                    or np.any(ids < 0) or np.any(ids >= len(centroids))):
                raise RuntimeError("HNSW returned invalid centroid IDs")
            chosen = self._canonical[ids[:, 0]]
            labels[begin:end] = chosen
            delta = points[begin:end].astype(np.float64) - self._centroids[chosen].astype(np.float64)
            distances[begin:end] = np.sum(delta * delta, axis=1, dtype=np.float64)
        elapsed = perf_counter() - start
        self._timings["query_s"] += elapsed
        self._timings["query_count"] += 1
        pending = self._pending_update_s
        self._pending_update_s = 0.0
        return AssignmentResult(labels, distances, elapsed, pending)
