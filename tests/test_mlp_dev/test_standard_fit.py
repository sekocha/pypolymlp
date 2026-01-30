"""Tests of standard fit."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.standard.fit import fit, fit_learning_curve, fit_standard

cwd = Path(__file__).parent


def test_fit(regdata_mp_149):
    """Test fit function from xtx and xty."""
    params, train = regdata_mp_149
    test = train
    model = fit(params, train, test)

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)
    assert model.alpha == pytest.approx(0.001)


def test_fit_standard(regdata_mp_149):
    """Test fit function from x and y."""
    params, train = regdata_mp_149
    test = train
    model = fit_standard(params, train, test)

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)
    assert model.alpha == pytest.approx(0.001)


def test_fit_learning_curve(regdata_mp_149):
    """Test fit function for learning curve."""
    params, train = regdata_mp_149
    test = train
    log = fit_learning_curve(params, train, test)
    nums = [l[0] for l in log]
    np.testing.assert_equal(nums, np.arange(1, 11) * 18)

    assert log[-3][1]["energy"] == pytest.approx(5.849589779404497e-06, rel=1e-2)
    assert log[-2][1]["energy"] == pytest.approx(5.841377969070585e-06, rel=1e-2)
    assert log[-1][1]["energy"] == pytest.approx(5.843977160697607e-06, rel=1e-2)
