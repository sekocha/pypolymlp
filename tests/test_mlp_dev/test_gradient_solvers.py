"""Tests of standard solvers."""

from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.gradient.solvers_cg import solver_cg

cwd = Path(__file__).parent


def test_solver_cg(dataxy_mp_149):
    """Test ridge solver using x and y."""
    x, y = dataxy_mp_149.x, dataxy_mp_149.y
    alpha = 1e-0
    coeffs = solver_cg(x=x, y=y, alpha=alpha, gtol=1e-4)
    np.testing.assert_allclose(coeffs[20], 0.02978607, rtol=1e-2)
    np.testing.assert_allclose(coeffs[30], -0.02865524, rtol=1e-2)


def test_solver_cg_xtx(dataxy_xtx_xty_mp_149):
    """Test ridge solver using xtx and xty."""
    xtx, xty = dataxy_xtx_xty_mp_149.xtx, dataxy_xtx_xty_mp_149.xty
    alpha = 1e-0
    coeffs = solver_cg(xtx=xtx, xty=xty, alpha=alpha, gtol=1e-4)
    np.testing.assert_allclose(coeffs[20], 0.02978607, rtol=1e-2)
    np.testing.assert_allclose(coeffs[30], -0.02865524, rtol=1e-2)
