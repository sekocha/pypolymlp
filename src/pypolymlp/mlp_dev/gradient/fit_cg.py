"""Functions for estimating regression coefficients using CG."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.gradient.solvers_cg import solver_cg


def _check_use_xy(polymlp: PolymlpDevCore):
    """Check whether xtx and xty data is used or not."""
    try:
        polymlp.check_memory_size_in_regression(use_gradient=True)
    except RuntimeError:
        return True
    return False


def fit_cg(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    gtol: float = 1e-3,
    max_iter: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients using CG."""
    polymlp = PolymlpDevCore(params, verbose=verbose)
    use_xy = _check_use_xy(polymlp)

    if use_xy:
        train_xy = polymlp.calc_xy(train)
    else:
        train_xy = polymlp.calc_xtx_xty(train)

    if max_iter is None:
        max_iter = polymlp.n_features * 10

    coefs, coef0 = [], None
    for alpha in reversed(polymlp.common_params.alphas):
        c = solver_cg(
            x=train_xy.x,
            y=train_xy.y,
            xtx=train_xy.xtx,
            xty=train_xy.xty,
            alpha=alpha,
            coef0=coef0,
            gtol=gtol,
            max_iter=max_iter,
            verbose=verbose,
        )
        coef0 = c
        coefs.append(c)
    coefs = np.array(coefs)[::-1].T

    rmse_train = polymlp.compute_rmse(coefs, train_xy, check_singular=True)
    train_xy.clear_data()

    if use_xy:
        test_xy = polymlp.calc_xy(
            test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
        )
    else:
        test_xy = polymlp.calc_xtx_xty(
            test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
        )

    rmse_test = polymlp.compute_rmse(coefs, test_xy)
    test_xy.clear_data()

    best_model = polymlp.get_best_model(
        coefs,
        train_xy.scales,
        rmse_train,
        rmse_test,
        train_xy.cumulative_n_features,
    )
    if verbose:
        polymlp.print_model_selection_log(rmse_train, rmse_test)

    return best_model
