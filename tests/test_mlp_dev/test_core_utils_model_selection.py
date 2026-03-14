"""Tests of utility functions for model selection."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.core.utils_model_selection import compute_rmse, get_best_model

cwd = Path(__file__).parent


def _assert(data_xy, params):
    """Assert rmse values."""
    coefs = np.ones((168, 3))
    coefs[:, 1] = 0.1
    coefs[:, 2] = 0.01
    rmse = compute_rmse(coefs, data_xy)

    rmse_train = rmse_test = rmse
    scales = data_xy.scales
    params_tmp = copy.deepcopy(params)
    params_tmp.alphas = (1e-3, 1e-2, 1e-1)
    model = get_best_model(params_tmp, coefs, scales, rmse_train, rmse_test)
    np.testing.assert_allclose(model.coeffs, coefs[:, 2])
    assert model.alpha == pytest.approx(0.1)


def test_compute_rmse_xy(dataxy_mp_149, regdata_mp_149):
    """Test for compute_rmse."""
    params, _ = regdata_mp_149
    _assert(dataxy_mp_149, params)


def test_compute_rmse_xtx_xty(dataxy_xtx_xty_mp_149, regdata_mp_149):
    """Test for compute_rmse using xtx and xty."""
    params, _ = regdata_mp_149
    _assert(dataxy_xtx_xty_mp_149, params)
