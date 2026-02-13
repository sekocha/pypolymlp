"""Tests of standard solvers."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.standard.solvers import solve_linear_equation, solver_ridge

cwd = Path(__file__).parent


def test_solve_ridge(dataxy_mp_149):
    """Test ridge solver using x and y."""
    x, y = dataxy_mp_149.x, dataxy_mp_149.y
    alphas = (1e-0, 1e1)
    coeffs = solver_ridge(x=x, y=y, alphas=alphas)
    np.testing.assert_allclose(coeffs[20], [0.02978377, 0.0064956], rtol=1e-3)
    np.testing.assert_allclose(coeffs[30], [-0.02865416, -0.00704682], rtol=1e-3)


def test_solve_ridge_xtx(dataxy_xtx_xty_mp_149):
    """Test ridge solver using xtx and xty."""
    xtx, xty = dataxy_xtx_xty_mp_149.xtx, dataxy_xtx_xty_mp_149.xty
    alphas = (1e-0, 1e1)
    coeffs = solver_ridge(xtx=xtx, xty=xty, alphas=alphas)
    np.testing.assert_allclose(coeffs[20], [0.02978377, 0.0064956], rtol=1e-3)
    np.testing.assert_allclose(coeffs[30], [-0.02865416, -0.00704682], rtol=1e-3)


def test_solve_linear_equation_xtx(dataxy_xtx_xty_mp_149):
    """Test solve_linear_equation."""
    xtx, xty = dataxy_xtx_xty_mp_149.xtx, dataxy_xtx_xty_mp_149.xty
    add = 10 * np.eye(xtx.shape[0])
    coeffs = solve_linear_equation(xtx + add, xty)
    assert coeffs[20] == pytest.approx(0.0064956, rel=1e-3)
    assert coeffs[30] == pytest.approx(-0.00704682, rel=1e-3)
