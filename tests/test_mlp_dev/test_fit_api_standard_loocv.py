"""Tests of standard fit."""

import copy
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.fit.api_fit import fit_polymlp

cwd = Path(__file__).parent


def test_fit(regdata_mp_149):
    """Test fit function from xtx and xty."""
    params, train = regdata_mp_149
    train2 = copy.deepcopy(train)
    test2 = copy.deepcopy(train)
    fit = fit_polymlp(params, train2, test2, use_cv=True, use_full_x=False)
    model = fit.best_model
    assert model.rmse_test == pytest.approx(0.0008210162371496754)

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)

    assert len(fit.all_models) == 5
