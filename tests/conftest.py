"""Small numerical fixtures; no imagery or GPU dependency."""

import numpy as np
import pytest


@pytest.fixture
def two_groups():
    points = np.array([[0, 0], [0, 2], [10, 10], [10, 12]], dtype=np.float32)
    initial = points[[0, 2]].copy()
    return points, initial


@pytest.fixture
def four_features():
    rng = np.random.default_rng(17)
    return np.ascontiguousarray(np.vstack([
        rng.normal(-2, 0.2, (24, 4)), rng.normal(2, 0.2, (24, 4)),
        rng.normal(6, 0.2, (24, 4)),
    ]), dtype=np.float32)
