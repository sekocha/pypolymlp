"""Tests of utility functions for model selection."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.core.utils_model_selection import (
    compute_rmse,
    get_best_model,
    smooth_alpha,
)

cwd = Path(__file__).parent


def _assert_rmse(data_xy):
    """Assert rmse values."""
    coefs = np.ones((168, 3))
    coefs[:, 1] = 0.1
    coefs[:, 2] = 0.01
    rmse = compute_rmse(coefs, data_xy)
    rmse_true = [70925.276143, 7114.733223, 733.720589]
    np.testing.assert_allclose(rmse, rmse_true, rtol=1e-5)

    rmse_train = rmse_test = rmse
    scales = data_xy.scales
    alphas = (1e-3, 1e-2, 1e-1)
    model = get_best_model(coefs, scales, alphas, rmse_train, rmse_test)
    np.testing.assert_allclose(model.coeffs, coefs[:, 2])
    assert model.alpha == pytest.approx(0.1)


def test_compute_rmse_xy(dataxy_mp_149):
    """Test for compute_rmse."""
    _assert_rmse(dataxy_mp_149)


def test_compute_rmse_xtx_xty(dataxy_xtx_xty_mp_149):
    """Test for compute_rmse using xtx and xty."""
    _assert_rmse(dataxy_xtx_xty_mp_149)


def test_smooth_alpha():
    """Test smooth_alpha."""
    n_data = 30
    alphas = 10 ** np.linspace(-5, 5, n_data)
    rmse = np.linspace(1, 0.1, n_data)
    rmse += np.pow(np.arange(n_data), 2) * 0.005
    best_alpha = smooth_alpha(alphas, rmse)
    assert best_alpha == pytest.approx(0.00011753158091642653)
