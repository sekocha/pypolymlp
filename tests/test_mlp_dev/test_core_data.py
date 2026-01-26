"""Tests of data_utils, data_standard, and data_sequential."""

import copy
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.data_sequential import calc_xtx_xty

cwd = Path(__file__).parent


def test_calc_xy(dataxy_mp_149):
    """Test for calc_xy."""
    data_xy = dataxy_mp_149
    assert data_xy.x.shape == (35820, 168)
    assert data_xy.y.shape[0] == 35820
    assert data_xy.weights.shape[0] == 35820
    assert data_xy.scales.shape[0] == 168

    idata, ifeature = 34500, 56
    assert data_xy.x[idata, ifeature] == pytest.approx(-1.6856145695486902, abs=1e-8)
    assert data_xy.y[idata] == pytest.approx(0.11092374)
    assert data_xy.weights[idata] == pytest.approx(1.0)
    assert data_xy.scales[ifeature] == pytest.approx(0.0032488632685359524)

    assert data_xy.min_energy == pytest.approx(-5.737324395625)
    assert data_xy.n_structures == 180
    assert data_xy.first_indices[0] == (0, 1260, 180)
    assert data_xy.cumulative_n_features is None


def _assert_xtx_xty(data_xy):
    """Assert data for xtx and xty in data_xy."""
    assert data_xy.xtx.shape == (168, 168)
    assert data_xy.xty.shape[0] == 168
    assert data_xy.scales.shape[0] == 168

    ifeature1, ifeature2 = 56, 59
    assert data_xy.xtx[ifeature1, ifeature2] == pytest.approx(
        1.7809598557388415e6, rel=1e-6
    )
    assert data_xy.xty[ifeature1] == pytest.approx(6.899130774433e5, rel=1e-6)
    assert data_xy.scales[ifeature1] == pytest.approx(0.0032488632685359524)

    assert data_xy.min_energy == pytest.approx(-5.737324395625)
    assert data_xy.total_n_data == 35820
    assert data_xy.first_indices is None
    assert data_xy.cumulative_n_features is None


def test_calc_xtx_xty1(dataxy_xtx_xty_mp_149):
    """Test for calc_xtx_xty."""
    _assert_xtx_xty(dataxy_xtx_xty_mp_149)


def test_calc_xtx_xty2(regdata_mp_149):
    """Test for parameters of calc_xtx_xty."""
    params, datasets = regdata_mp_149
    datasets_ = copy.deepcopy(datasets)
    data_xy = calc_xtx_xty(params, datasets_, batch_size=10)
    _assert_xtx_xty(data_xy)

    data_xy = calc_xtx_xty(params, datasets_, n_features_threshold=10)
    _assert_xtx_xty(data_xy)

    data_xy = calc_xtx_xty(params, datasets_, batch_size=50, use_gradient=True)
    _assert_xtx_xty(data_xy)
