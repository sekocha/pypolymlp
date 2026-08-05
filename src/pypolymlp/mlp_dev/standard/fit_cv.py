"""Functions for estimating regression coefficients from datasets."""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore


def fit_cv(
    params: PolymlpParams,
    train: DatasetList,
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients using the cross-validation.

    Parameters
    ----------
    params: Parameters of polymlp.
    train: Training datasets.
    batch_size: Batch size for sequential regression.
                If None, the batch size is automatically determined
                depending on the memory size and number of features.
    """
    polymlp = PolymlpDevCore(params, verbose=verbose)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xtx_xty(train, batch_size=batch_size)

    xtx = train_xy.xtx
    xty = train_xy.xty
    alphas = params.alphas
    if verbose:
        print("Regression:", flush=True)

    cv_scores = []
    n_features = xtx.shape[0]
    coefs_array = np.zeros((n_features, len(alphas)))
    alpha_prev = 0.0
    for i, alpha in enumerate(alphas):
        if verbose:
            print("- alpha:", alpha, flush=True)
        add = alpha - alpha_prev
        if verbose:
            print("  Compute X.T @ X + alpha @ I", flush=True)
        xtx.flat[:: n_features + 1] += add
        if verbose:
            print("  Compute inverse matrix", flush=True)
        inv_xtx = np.linalg.inv(xtx)
        coefs_single = inv_xtx @ xty
        coefs_array[:, i] = coefs_single

        rmse_cv = polymlp.compute_rmse_cv(
            coefs_single,
            train_xy.scales,
            train,
            inv_xtx,
            batch_size=20,
        )
        cv_scores.append(rmse_cv)
        alpha_prev = alpha

    if verbose:
        polymlp.print_model_selection_log([0] * len(cv_scores), cv_scores)

    best_model = polymlp.get_best_model(
        coefs_array,
        train_xy.scales,
        [0] * len(cv_scores),
        cv_scores,
        train_xy.cumulative_n_features,
    )

    xtx.flat[:: n_features + 1] -= alpha
    xtx.flat[:: n_features + 1] += best_model.alpha
    inv_xtx = np.linalg.inv(xtx)

    return best_model, inv_xtx
