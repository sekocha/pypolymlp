"""Tests of utility functions for applying weights."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.core.utils import collect_observations
from pypolymlp.mlp_dev.core.utils_weights import apply_weights, apply_weights_overwrite

cwd = Path(__file__).parent


def _init(regdata_mp_149, dataxy_mp_149):
    """Initialize."""
    params, datasets = regdata_mp_149
    first_indices = dataxy_mp_149.first_indices[0]
    x = copy.deepcopy(dataxy_mp_149.x)
    y = collect_observations(datasets[0], x.shape[0], first_indices)
    return x, y, datasets[0], first_indices


def _init_ow(regdata_mp_149, dataxy_mp_149):
    """Initialize."""
    params, datasets = regdata_mp_149
    first_indices = dataxy_mp_149.first_indices[0]
    x = copy.deepcopy(dataxy_mp_149.x)
    y = np.zeros(x.shape[0])
    w = np.ones(x.shape[0])
    return x, y, w, datasets[0], first_indices


def test_apply_weights1(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights."""
    x, y, data, first_indices = _init(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights(x, y, data, first_indices)
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740


def test_apply_weights2(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights."""
    x, y, data, first_indices = _init(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights(x, y, data, first_indices, use_auto_weight=False)
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740


def test_apply_weights3(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights."""
    x, y, data, first_indices = _init(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights(
        x, y, data, first_indices, weight_force=0.5, weight_stress=0.2
    )
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 180
    assert np.count_nonzero(np.isclose(w, 0.5)) == 34560
    assert np.count_nonzero(np.isclose(w, 0.2)) == 0


def test_apply_weights4(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights."""
    x, y, data, first_indices = _init(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights(
        x,
        y,
        data,
        first_indices,
        weight_force=0.5,
        weight_stress=0.2,
        use_auto_weight=False,
    )
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 180
    assert np.count_nonzero(np.isclose(w, 0.5)) == 34560
    assert np.count_nonzero(np.isclose(w, 0.2)) == 0


def test_apply_weights_overwrite1(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights_overwrite."""
    x, y, w, data, first_indices = _init_ow(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights_overwrite(x, y, w, data, first_indices)
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740


def test_apply_weights_overwrite2(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights_overwrite."""
    x, y, w, data, first_indices = _init_ow(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights_overwrite(
        x, y, w, data, first_indices, use_auto_weight=False
    )
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740


def test_apply_weights_overwrite3(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights_overwrite."""
    x, y, w, data, first_indices = _init_ow(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights_overwrite(
        x, y, w, data, first_indices, weight_force=0.5, weight_stress=0.2
    )
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 180
    assert np.count_nonzero(np.isclose(w, 0.5)) == 34560
    assert np.count_nonzero(np.isclose(w, 0.2)) == 0


def test_apply_weights_overwrite4(regdata_mp_149, dataxy_mp_149):
    """Test for apply_weights_overwrite."""
    x, y, w, data, first_indices = _init_ow(regdata_mp_149, dataxy_mp_149)
    x, y, w = apply_weights_overwrite(
        x,
        y,
        w,
        data,
        first_indices,
        use_auto_weight=False,
        weight_force=0.5,
        weight_stress=0.2,
    )
    np.testing.assert_allclose(y[:3], [-3.67121086e02, -3.67115343e02, -3.67102616e02])
    assert np.count_nonzero(np.isclose(w, 1.0)) == 180
    assert np.count_nonzero(np.isclose(w, 0.5)) == 34560
    assert np.count_nonzero(np.isclose(w, 0.2)) == 0
