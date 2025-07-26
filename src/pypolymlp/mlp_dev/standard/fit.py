"""Functions for estimating regression coefficients from datasets."""

from typing import Optional, Union

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.standard.model_selection import (
    check_singular,
    compute_rmse,
    get_best_model,
    print_log,
)
from pypolymlp.mlp_dev.standard.solvers import solver_ridge

# from pypolymlp.mlp_dev.standard.learning_curve import LearningCurve


def fit(
    # def fit_standard(
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

    train_xy = polymlp.calc_xtx_xty(train, batch_size=batch_size)
    coefs = solver_ridge(
        xtx=train_xy.xtx,
        xty=train_xy.xty,
        alphas=common_params.alphas,
        verbose=verbose,
    )
    rmse_train = compute_rmse(train_xy, coefs)
    check_singular(rmse_train)
    train_xy.clear_data()

    test_xy = polymlp.calc_xtx_xty(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
        batch_size=batch_size,
    )
    rmse_test = compute_rmse(test_xy, coefs)
    test_xy.clear_data()

    best_model = get_best_model(
        coefs, train_xy.scales, common_params.alphas, rmse_train, rmse_test
    )

    if verbose:
        print_log(rmse_train, rmse_test, common_params.alphas)

    return best_model


def fit_standard(
    # def fit(
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients with direct evaluation of X."""

    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xy(train)
    coefs = solver_ridge(
        x=train_xy.x,
        y=train_xy.y,
        alphas=common_params.alphas,
        verbose=verbose,
    )
    rmse_train = compute_rmse(train_xy, coefs)
    check_singular(rmse_train)
    train_xy.clear_data()

    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )
    rmse_test = compute_rmse(test_xy, coefs)
    test_xy.clear_data()

    best_model = get_best_model(
        coefs, train_xy.scales, common_params.alphas, rmse_train, rmse_test
    )
    if verbose:
        print_log(rmse_train, rmse_test, common_params.alphas)

    return best_model


def fit_learning_curve(
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    verbose: bool = False,
):
    """Calculate learning curve."""

    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    if len(train) > 1:
        raise RuntimeError("Use single dataset for learning curve calculation")

    return None


#     polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
#     total_n_atoms = polymlp._train[0].total_n_atoms
#
#     learning = LearningCurve(polymlp, total_n_atoms, verbose=verbose)
#     learning.run()
#     return learning
