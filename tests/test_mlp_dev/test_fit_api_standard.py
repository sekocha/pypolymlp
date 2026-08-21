"""Tests of standard fit."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.fit.api_fit import fit_learning_curve, fit_polymlp

cwd = Path(__file__).parent


def test_fit(regdata_mp_149):
    """Test fit function from xtx and xty."""
    params, train = regdata_mp_149
    train2 = copy.deepcopy(train)
    test2 = copy.deepcopy(train)
    fit = fit_polymlp(params, train2, test2, use_full_x=False)
    model = fit.best_model

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)

    assert len(fit.all_models) == 5


def test_fit_standard(regdata_mp_149):
    """Test fit function from x and y."""
    params, train = regdata_mp_149
    train2 = copy.deepcopy(train)
    test2 = copy.deepcopy(train)
    fit = fit_polymlp(params, train2, test2, use_full_x=True)
    model = fit.best_model

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)

    assert len(fit.all_models) == 5


def test_fit_learning_curve(regdata_mp_149):
    """Test fit function for learning curve."""
    params, train = regdata_mp_149
    train2 = copy.deepcopy(train)
    test2 = copy.deepcopy(train)
    params.alphas = [1e2, 1e3, 1e4]
    fit = fit_learning_curve(params, train2, test2)
    log = fit.error_log
    params.alphas = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    nums = [l[0] for l in log]
    np.testing.assert_equal(nums, np.arange(1, 11) * 18)

    assert log[-3][1]["energy"] == pytest.approx(2.806378176050595e-06, rel=1e-2)
    assert log[-2][1]["energy"] == pytest.approx(2.7852516328822455e-06, rel=1e-2)
    assert log[-1][1]["energy"] == pytest.approx(2.7604438411037103e-06, rel=1e-2)
