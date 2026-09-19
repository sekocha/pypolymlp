"""Tests of online fit."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.fit.api_fit import fit_polymlp_online

cwd = Path(__file__).parent


def test_fit_adam(regdata_mp_149):
    """Test fit function using online from xtx and xty."""
    params, train = regdata_mp_149
    coeffs = np.zeros(168)
    train2 = copy.deepcopy(train)
    fit = fit_polymlp_online(params, train2, coeffs=coeffs, n_epochs=3)
    model = fit.best_model

    assert len(model.scaled_coeffs) == 168
