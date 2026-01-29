"""Functions for estimating regression coefficients from datasets."""

from typing import Optional, Union

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.dataset import DatasetList
from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore, eval_accuracy
from pypolymlp.mlp_dev.standard.solvers import solver_ridge
from pypolymlp.mlp_dev.standard.utils_learning_curve import print_learning_curve_log


def fit(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: DatasetList,
    test: DatasetList,
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients without computing entire X.

    Parameters
    ----------
    params: Parameters of polymlp.
    train: Training datasets.
    test: Test datasets.
    batch_size: Batch size for sequential regression.
                If None, the batch size is automatically determined
                depending on the memory size and number of features.
    """
    polymlp = PolymlpDevCore(params, verbose=verbose)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xtx_xty(train, batch_size=batch_size)
    coefs = solver_ridge(
        xtx=train_xy.xtx,
        xty=train_xy.xty,
        alphas=polymlp.common_params.alphas,
        verbose=verbose,
    )
    rmse_train = polymlp.compute_rmse(coefs, train_xy, check_singular=True)
    train_xy.clear_data()

    test_xy = polymlp.calc_xtx_xty(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
        batch_size=batch_size,
    )
    rmse_test = polymlp.compute_rmse(coefs, test_xy)
    test_xy.clear_data()

    if verbose:
        polymlp.print_model_selection_log(rmse_train, rmse_test)

    best_model = polymlp.get_best_model(
        coefs,
        train_xy.scales,
        rmse_train,
        rmse_test,
        train_xy.cumulative_n_features,
    )

    return best_model


def fit_standard(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: DatasetList,
    test: DatasetList,
    verbose: bool = False,
):
    """Estimate MLP coefficients with direct evaluation of X.

    Parameters
    ----------
    params: Parameters of polymlp.
    train: Training datasets.
    test: Test datasets.
    """

    polymlp = PolymlpDevCore(params, verbose=verbose)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xy(train)
    coefs = solver_ridge(
        x=train_xy.x,
        y=train_xy.y,
        alphas=polymlp.common_params.alphas,
        verbose=verbose,
    )
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


def fit_learning_curve(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: DatasetList,
    test: DatasetList,
    verbose: bool = False,
):
    """Calculate learning curve.

    Parameters
    ----------
    params: Parameters of polymlp.
    train: Training datasets.
    test: Test datasets.
    """
    if len(train) != 1:
        raise RuntimeError(
            "Number of training datasets must be one for learning curve."
        )

    polymlp = PolymlpDevCore(params, verbose=verbose)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xy(train)
    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )

    if verbose:
        print("Calculate learning curve.", flush=True)

    error_log = []
    n_train = train_xy.n_structures
    for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
        if verbose:
            print("------------- n_samples:", n_samples, "-------------", flush=True)

        x, y = train_xy.slice(n_samples, train[0].total_n_atoms)
        coefs = solver_ridge(
            x=x,
            y=y,
            alphas=polymlp.common_params.alphas,
            verbose=False,
        )
        rmse_train = polymlp.compute_rmse(coefs, x=x, y=y)
        rmse_test = polymlp.compute_rmse(coefs, test_xy)
        best_model = polymlp.get_best_model(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        if verbose:
            polymlp.print_model_selection_log(rmse_train, rmse_test)

        error = eval_accuracy(best_model, test, log_energy=False, tag="test")
        for val in error.values():
            error_log.append([n_samples, val])

    if verbose:
        print_learning_curve_log(error_log)

    return error_log
