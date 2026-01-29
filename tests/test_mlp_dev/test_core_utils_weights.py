"""Tests of utility functions for applying weights."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.core.utils_weights import apply_weights

cwd = Path(__file__).parent


def test_apply_weights(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights."""
    params, datasets = regdata_mp_149
    first_indices = dataxy_mp_149.first_indices[0]
    x = copy.deepcopy(dataxy_mp_149.x)
    y = np.zeros(x.shape[0])
    w = np.ones(x.shape[0])
    x, y, w = apply_weights(x, y, w, datasets[0], first_indices)
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740
