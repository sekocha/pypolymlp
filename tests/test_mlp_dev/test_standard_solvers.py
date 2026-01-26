"""Tests of standard solvers."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.standard.solvers import solve_linear_equation, solver_ridge

cwd = Path(__file__).parent


def test_solve_ridge(dataxy_mp_149):
    """Test ridge solver using x and y."""
    x, y = dataxy_mp_149.x, dataxy_mp_149.y
    alphas = (1e-1, 1e-0)
    coeffs = solver_ridge(x=x, y=y, alphas=alphas)
    np.testing.assert_allclose(coeffs[20], [0.08227511, 0.02978607], rtol=1e-4)
    np.testing.assert_allclose(coeffs[30], [-0.06958063, -0.02865442], rtol=1e-4)


def test_solve_ridge_xtx(dataxy_xtx_xty_mp_149):
    """Test ridge solver using xtx and xty."""
    xtx, xty = dataxy_xtx_xty_mp_149.xtx, dataxy_xtx_xty_mp_149.xty
    alphas = (1e-1, 1e-0)
    coeffs = solver_ridge(xtx=xtx, xty=xty, alphas=alphas)
    np.testing.assert_allclose(coeffs[20], [0.08228079, 0.02978955], rtol=1e-4)
    np.testing.assert_allclose(coeffs[30], [-0.06970561, -0.02865524], rtol=1e-4)


def test_solve_linear_equation_xtx(dataxy_xtx_xty_mp_149):
    """Test solve_linear_equation."""
    xtx, xty = dataxy_xtx_xty_mp_149.xtx, dataxy_xtx_xty_mp_149.xty
    add = 0.001 * np.eye(xtx.shape[0])
    coeffs = solve_linear_equation(xtx + add, xty)
    assert coeffs[20] == pytest.approx(-0.17924954891801606)
    assert coeffs[30] == pytest.approx(-0.5097204482523142)
