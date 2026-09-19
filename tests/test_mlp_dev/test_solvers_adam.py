"""Tests of adam solvers."""

from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.fit.solvers_adam import solver_adam

cwd = Path(__file__).parent


def test_solver_adam(dataxy_mp_149):
    """Test adam solver using x and y."""
    x, y = dataxy_mp_149.x, dataxy_mp_149.y
    coef0 = np.ones(x.shape[1])
    coeffs = solver_adam(x=x, y=y, coef0=coef0, n_epochs=3, gtol=1e-4)
    assert coeffs.shape[0] == coef0.shape[0]
