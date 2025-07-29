"""Functions sgd for estimating regression coefficients from datasets."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams

# from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore, eval_accuracy
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.gradient.solvers_sgd import solver_sgd
from pypolymlp.mlp_dev.standard.utils_model_selection import (
    compute_rmse,
    get_best_model,
    print_log,
)


def fit_sgd(
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients without computing entire X.

    Parameters
    ----------
    batch_size: Batch size for sequential regression.
                If None, the batch size is automatically determined
                depending on the memory size and number of features.
    """
    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xy(train)
    coefs = []
    # coef0 = None
    for alpha in common_params.alphas:
        c = solver_sgd(train_xy.x, train_xy.y, alpha=alpha, gtol=1e-2, verbose=verbose)
        # coef0 = c
        coefs.append(c)
    coefs = np.array(coefs).T

    rmse_train = compute_rmse(coefs, train_xy, check_singular=True)
    train_xy.clear_data()
    print(rmse_train)

    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )
    rmse_test = compute_rmse(coefs, test_xy)
    print(rmse_test)
    test_xy.clear_data()

    best_model = get_best_model(
        coefs,
        train_xy.scales,
        common_params.alphas,
        rmse_train,
        rmse_test,
        params,
        train_xy.cumulative_n_features,
    )
    if verbose:
        print_log(rmse_train, rmse_test, common_params.alphas)

    return best_model
