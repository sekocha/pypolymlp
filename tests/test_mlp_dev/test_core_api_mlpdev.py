"""Tests of data_utils, data_standard, and data_sequential."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore

cwd = Path(__file__).parent


def test_api_functions(regdata_mp_149):
    """Test for functions in API."""
    params, _ = regdata_mp_149

    core = PolymlpDevCore(params, use_gradient=False)
    f_attr, p_attr, a_attr = core.get_features_attr()
    assert isinstance(f_attr, list) == True
    assert isinstance(p_attr, list) == True
    assert isinstance(a_attr, dict) == True
    mem = core.check_memory_size_in_regression()
    assert mem == pytest.approx(0.0)

    assert core.n_features == 168
    assert core.params == params
    assert core.common_params == params
    assert core.is_hybrid == False

    core.params = [params, params]
    assert core.is_hybrid == True
    assert core.n_features == 336


def _assert_rmse(core, data_xy):
    """Assert rmse values."""
    coefs = np.ones((168, 3))
    coefs[:, 1] = 0.1
    coefs[:, 2] = 0.01
    rmse = core.compute_rmse(coefs, data_xy)
    rmse_true = [70925.276143, 7114.733223, 733.720589]
    np.testing.assert_allclose(rmse, rmse_true, rtol=1e-5)

    rmse_train = rmse_test = rmse
    scales = data_xy.scales
    model = core.get_best_model(coefs, scales, rmse_train, rmse_test)
    np.testing.assert_allclose(model.coeffs, coefs[:, 2])
    assert model.alpha == pytest.approx(0.1)


def test_api_calc_xy(regdata_mp_149):
    """Test for x and y calculation functions in API."""
    params, datasets_ = regdata_mp_149
    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xy(datasets_)
    assert data_xy.x.shape == (35820, 168)

    _assert_rmse(core, data_xy)


def test_api_calc_xtx_xty1(regdata_mp_149):
    """Test for xtx and xty calculation functions in API."""
    params, datasets = regdata_mp_149
    datasets_ = copy.deepcopy(datasets)
    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xtx_xty(datasets_)
    assert data_xy.xtx.shape == (168, 168)

    _assert_rmse(core, data_xy)


def test_api_calc_xtx_xty2(regdata_mp_149):
    """Test for xtx and xty calculation functions in API."""
    params, datasets = regdata_mp_149
    datasets_ = copy.deepcopy(datasets)
    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xtx_xty(datasets_, batch_size=10)
    assert data_xy.xtx.shape == (168, 168)

    _assert_rmse(core, data_xy)
