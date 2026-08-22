"""Functions for model selection."""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.data_sequential import compute_features_single_batch
from pypolymlp.mlp_dev.core.data_utils import PolymlpDataXY
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP
from pypolymlp.mlp_dev.core.features_attr import get_num_features
from pypolymlp.mlp_dev.core.utils_sequential import get_auto_batch_size, get_batch_slice


def _check_singular(rmse: np.ndarray, error_threshold: float = 1e6):
    """Check whether X.T @ X + penalty is ill-defined."""
    if np.all(np.array(rmse) > error_threshold):
        raise RuntimeError(
            "Matrix (X.T @ X + alpha * I) may be singular. "
            "This singularity issue might be reduced by increasing "
            "the value of alpha (the magnitude of the penalty term)."
        )


def compute_rmse_standard(
    coefs_array: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    check_singular: bool = False,
):
    """Compute RMSEs from x and y."""
    pred = (x @ coefs_array).T
    rmse_array = np.array([rmse(y, p) for p in pred])
    if check_singular:
        _check_singular(rmse_array)
    return rmse_array


def compute_mse(
    xtx: np.ndarray,
    xty: np.ndarray,
    y_sq_norm,
    size: int,
    coefs: np.ndarray,
):
    """Compute mean squared error from xtx and xty."""
    v1 = coefs @ (xtx @ coefs)
    v2 = -2 * coefs @ xty
    return (v1 + v2 + y_sq_norm) / size


def compute_rmse_seq(
    coefs_array: np.ndarray,
    data_xy: PolymlpDataXY,
    check_singular: bool = False,
):
    """Compute RMSEs from xtx and xty."""
    rmse_array = []
    for coefs in coefs_array.T:
        mse = compute_mse(
            data_xy.xtx,
            data_xy.xty,
            data_xy.y_sq_norm,
            data_xy.total_n_data,
            coefs,
        )
        try:
            rmse_array.append(np.sqrt(mse))
        except:
            rmse_array.append(1e10)

    if check_singular:
        _check_singular(rmse_array)
    return np.array(rmse_array)


def compute_rmse(
    coefs_array: np.ndarray,
    data_xy: Optional[PolymlpDataXY] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    check_singular: bool = False,
):
    """Compute RMSEs."""
    if data_xy is not None:
        if data_xy.xtx is None:
            return compute_rmse_standard(coefs_array, data_xy.x, data_xy.y)
        return compute_rmse_seq(coefs_array, data_xy)
    else:
        return compute_rmse_standard(coefs_array, x, y)


def compute_rmse_cv(
    params: PolymlpParams,
    coefs_single: np.ndarray,
    scales: np.ndarray,
    datasets: DatasetList,
    inv_xtx: np.ndarray,
    verbose: bool = False,
    batch_size: int = 20,
):
    """Compute weighted RMSEs using the cross-validation method."""
    n_features = get_num_features(params)
    if batch_size is None:
        batch_size = get_auto_batch_size(
            n_features,
            verbose=verbose,
        )

    sum_se = 0
    num_comp = 0
    for data in datasets:
        if verbose:
            print("----- Dataset:", data.name, "-----", flush=True)
        # data.sort_dft()
        n_str = len(data.structures)
        begin_ids, end_ids = get_batch_slice(n_str, batch_size)
        for begin, end in zip(begin_ids, end_ids):
            if verbose:
                print("Structures:", end, "/", n_str, flush=True)
            sliced_data = data.slice_dft(begin, end)
            x, x_w, x_w2, y_w, _ = compute_features_single_batch(
                params,
                sliced_data,
                verbose=verbose,
            )
            x = x / scales
            x_w = x_w / scales
            x_w2 = x_w2 / scales

            if verbose:
                print(
                    f" Matrix shape (Projection matrix) = ({x.shape[0]} {x.shape[0]})"
                )
                print(" Compute X @ (X.T @ X + alpha @ I)^-1 @ X.T")
            hat_h = x @ inv_xtx @ x_w2.T
            hat_h_diag = -np.diagonal(hat_h)
            hat_h_diag += 1

            y_pred = np.dot(x_w, coefs_single)
            sum_se += np.sum(((y_pred - y_w) / hat_h_diag) ** 2)
            num_comp += y_pred.shape[0]

    return np.sqrt(sum_se / num_comp)


def get_best_model(
    params: PolymlpParams,
    coefs_array: np.ndarray,
    scales: np.ndarray,
    rmse_train: np.ndarray,
    rmse_test: np.ndarray,
    cumulative_n_features: Optional[tuple] = None,
):
    """Return best polymlp model."""
    if len(params.alphas) != coefs_array.shape[1]:
        raise RuntimeError("Shapes of alphas and coeffs not consistent.")
    if len(params.alphas) != len(rmse_train):
        raise RuntimeError("Shapes of alphas and rmse_train not consistent.")
    if len(params.alphas) != len(rmse_test):
        raise RuntimeError("Shapes of alphas and rmse_test not consistent.")

    idx = np.argmin(rmse_test)
    best_model = PolymlpDataMLP(
        coeffs=coefs_array[:, idx],
        scales=scales,
        rmse_train=rmse_train[idx],
        rmse_test=rmse_test[idx],
        alpha=params.alphas[idx],
        params=params,
        cumulative_n_features=cumulative_n_features,
    )
    return best_model


def get_all_models(
    params: PolymlpParams,
    coefs_array: np.ndarray,
    scales: np.ndarray,
    rmse_train: np.ndarray,
    rmse_test: np.ndarray,
    cumulative_n_features: Optional[tuple] = None,
):
    """Return best polymlp model."""
    if len(params.alphas) != coefs_array.shape[1]:
        raise RuntimeError("Shapes of alphas and coeffs not consistent.")
    if len(params.alphas) != len(rmse_train):
        raise RuntimeError("Shapes of alphas and rmse_train not consistent.")
    if len(params.alphas) != len(rmse_test):
        raise RuntimeError("Shapes of alphas and rmse_test not consistent.")

    all_models = []
    for idx in range(coefs_array.shape[1]):
        model = PolymlpDataMLP(
            coeffs=coefs_array[:, idx],
            scales=scales,
            rmse_train=rmse_train[idx],
            rmse_test=rmse_test[idx],
            alpha=params.alphas[idx],
            params=params,
            cumulative_n_features=cumulative_n_features,
        )
        all_models.append(model)
    return all_models


def print_log(
    params: PolymlpParams,
    rmse_train: np.ndarray,
    rmse_test: np.ndarray,
    use_cv: bool = False,
    error_threshold: float = 1e6,
):
    """Output log for ridge regression."""
    print("Regression: model selection ...", flush=True)
    str1 = ": rmse (train), cv =" if use_cv else ": rmse (train, test) ="
    for a, rmse1, rmse2 in zip(params.alphas, rmse_train, rmse_test):
        if rmse1 > error_threshold:
            text = str1 + "Failed, Failed"
            print("- alpha =", "{:.3e}".format(a), text, flush=True)
        else:
            print(
                "- alpha =",
                "{:.3e}".format(a),
                str1,
                "{:.5f}".format(rmse1),
                "{:.5f}".format(rmse2),
                flush=True,
            )
