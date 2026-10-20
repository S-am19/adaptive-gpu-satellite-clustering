"""CUDA-independent public contracts and strict host-array validation."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import numpy as np


def validate_integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass(frozen=True)
class FitConfig:
    k: int
    seed: int
    max_steps: int
    tolerance: float
    batch_size: int | None = None
    probe_interval: int = 10
    ann_effort: int | None = None

    def __post_init__(self):
        for name in ("k", "max_steps", "probe_interval"):
            validate_integer(name, getattr(self, name))
        validate_integer("seed", self.seed, minimum=0)
        for name in ("batch_size", "ann_effort"):
            if getattr(self, name) is not None:
                validate_integer(name, getattr(self, name))
        if isinstance(self.tolerance, (bool, np.bool_)) or not isinstance(self.tolerance, Real):
            raise TypeError("tolerance must be a real number")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")


@dataclass
class AssignmentResult:
    labels: object
    squared_distances: object
    query_time_s: float
    index_update_time_s: float = 0.0


@dataclass
class ClusterStats:
    sums: object
    counts: object


class AssignmentBackend(Protocol):
    def prepare(self, centroids, effort=None): ...
    def assign(self, points, centroids, effort=None) -> AssignmentResult: ...
    def refresh(self, new_centroids): ...


@dataclass
class RunResult:
    centroids: np.ndarray
    labels: np.ndarray
    history: list[dict]
    timings: dict
    metadata: dict

    @property
    def sse(self) -> float:
        """Final exact full-data SSE, independent of optional/probe history.

        Producers must set metadata['final_exact_sse'] after a full exact
        assignment at the returned centroids. Missing/invalid values fail
        explicitly; an intermediate history objective is never a fallback.
        """
        value = self.metadata.get("final_exact_sse")
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError("metadata['final_exact_sse'] must be a finite nonnegative scalar")
        return float(value)


def validate_features(name, array):
    """Reject implicit casts/copies at public CPU entry points."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must have nonempty shape [rows, features]")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def validate_points_centroids(points, centroids):
    validate_features("points", points)
    validate_features("centroids", centroids)
    if points.shape[1] != centroids.shape[1]:
        raise ValueError("points and centroids must have the same feature dimension")
    if centroids.shape[0] > np.iinfo(np.int32).max:
        raise ValueError("centroid count exceeds int32 label capacity")
