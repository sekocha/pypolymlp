"""Functions sgd for estimating regression coefficients from datasets."""

from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.gradient.solvers_sgd import solver_sgd


def fit_sgd(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    verbose: bool = False,
):
    """Estimate MLP coefficients without computing entire X."""

    polymlp = PolymlpDevCore(params)

    train_xy = polymlp.calc_xy(train)
    coefs = []
    for alpha in polymlp.common_params.alphas:
        c = solver_sgd(train_xy.x, train_xy.y, alpha=alpha, gtol=1e-2, verbose=verbose)
        coefs.append(c)
    coefs = np.array(coefs).T

    rmse_train = polymlp.compute_rmse(coefs, train_xy, check_singular=True)
    train_xy.clear_data()
    print(rmse_train)

    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )
    rmse_test = polymlp.compute_rmse(coefs, test_xy)
    print(rmse_test)
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
