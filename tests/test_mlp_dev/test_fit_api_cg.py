"""Tests of cg fit."""

import copy
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.fit.api_fit import fit_polymlp

cwd = Path(__file__).parent


def test_fit_cg(regdata_mp_149):
    """Test fit function from xtx and xty."""
    params, train = regdata_mp_149
    train2 = copy.deepcopy(train)
    test2 = copy.deepcopy(train)
    fit = fit_polymlp(params, train2, test2, use_cg=True)
    model = fit.best_model

    model.scaled_coeffs[0] == pytest.approx(-6.40229659e02)
    model.scaled_coeffs[1] == pytest.approx(1.73844624e05)

    assert len(fit.all_models) == 5
