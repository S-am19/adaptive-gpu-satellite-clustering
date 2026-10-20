"""Reproducible initialization shared by compatible algorithm variants."""

import numpy as np

from .contracts import validate_features, validate_integer


def initialize_centroids(X, k, *, seed):
    """Sample distinct row indices (not necessarily distinct values)."""
    validate_features("X", X)
    validate_integer("k", k)
    validate_integer("seed", seed, minimum=0)
    if k > X.shape[0] or k > np.iinfo(np.int32).max:
        raise ValueError("k must not exceed N or int32 label capacity")
    indices = np.random.default_rng(seed).choice(X.shape[0], size=k, replace=False)
    return np.ascontiguousarray(X[indices]).copy()
