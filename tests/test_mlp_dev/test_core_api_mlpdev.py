"""Tests of data_utils, data_standard, and data_sequential."""

import copy
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore

cwd = Path(__file__).parent


def test_api_calc(regdata_mp_149):
    """Test for calculation functions in API."""
    params, datasets = regdata_mp_149
    datasets_ = copy.deepcopy(datasets)

    core = PolymlpDevCore(params, use_gradient=False)
    data_xy = core.calc_xy(datasets_)
    assert data_xy.x.shape == (35820, 168)
    data_xy = core.calc_xtx_xty(datasets_)
    assert data_xy.xtx.shape == (168, 168)
    data_xy = core.calc_xtx_xty(datasets_, batch_size=10)
    assert data_xy.xtx.shape == (168, 168)


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
