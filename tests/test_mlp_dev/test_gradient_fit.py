"""Tests of gradient fit."""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.gradient.fit_cg import fit_cg

cwd = Path(__file__).parent


def test_fit_cg(regdata_mp_149):
    """Test fit function from xtx and xty."""
    params, train = regdata_mp_149
    test = train
    model = fit_cg(params, train, test)

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)
    assert model.alpha == pytest.approx(0.001)
