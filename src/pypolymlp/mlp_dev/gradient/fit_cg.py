"""Functions for estimating regression coefficients using CG."""

from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.gradient.solvers_cg import solver_cg


def fit_cg(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    verbose: bool = False,
):
    """Estimate MLP coefficients using CG."""
    polymlp = PolymlpDevCore(params)

    train_xy = polymlp.calc_xy(train)
    coefs, coef0 = [], None
    max_iter = train_xy.x.shape[0] * 5
    for alpha in polymlp.common_params.alphas:
        c = solver_cg(
            train_xy.x,
            train_xy.y,
            alpha=alpha,
            coef0=coef0,
            gtol=1e-2,
            max_iter=max_iter,
            verbose=verbose,
        )
        coef0 = c
        coefs.append(c)
    coefs = np.array(coefs).T

    rmse_train = polymlp.compute_rmse(coefs, train_xy, check_singular=True)
    train_xy.clear_data()

    test_xy = polymlp.calc_xy(
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
