"""Functions for estimating regression coefficients from datasets."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
from pypolymlp.mlp_dev.standard.model_selection import compute_rmse
from pypolymlp.mlp_dev.standard.solvers import solver_ridge

# from pypolymlp.mlp_dev.standard.learning_curve import LearningCurve


def _check_singular(rmse: np.ndarray, error_threshold: float = 1e6):
    """Check whether X.T @ X + penalty is ill-defined."""
    if np.all(np.array(rmse) > error_threshold):
        raise RuntimeError(
            "Matrix (X.T @ X + alpha * I) may be singular. "
            "This singularity issue might be reduced by increasing "
            "the value of alpha (the magnitude of the penalty term)."
        )


def _print_log(
    rmse_train: np.ndarray,
    rmse_test: np.ndarray,
    alphas: np.ndarray,
    error_threshold: float = 1e6,
):
    """Output log for ridge regression."""
    print("Regression: model selection ...", flush=True)
    for a, rmse1, rmse2 in zip(alphas, rmse_train, rmse_test):
        if rmse1 > error_threshold:
            text = ": rmse (train, test) = Failed, Failed"
            print("- alpha =", "{:.3e}".format(a), text, flush=True)
        else:
            print(
                "- alpha =",
                "{:.3e}".format(a),
                ": rmse (train, test) =",
                "{:.5f}".format(rmse1),
                "{:.5f}".format(rmse2),
                flush=True,
            )


# def fit(
def fit_standard(
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
    _check_singular(rmse_train)
    train_xy.clear_data()

    test_xy = polymlp.calc_xtx_xty(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
        batch_size=batch_size,
    )
    rmse_test = compute_rmse(test_xy, coefs)
    test_xy.clear_data()

    if verbose:
        _print_log(rmse_train, rmse_test, common_params.alphas)

    return None


# def fit_standard(
def fit(
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
    _check_singular(rmse_train)
    train_xy.clear_data()

    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )
    rmse_test = compute_rmse(test_xy, coefs)
    test_xy.clear_data()

    if verbose:
        _print_log(rmse_train, rmse_test, common_params.alphas)

    return None


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
