"""Functions for model selection."""

import numpy as np

from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.data import PolymlpDataXY
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP


def compute_rmse_standard(data_xy: PolymlpDataXY, coefs_array: np.ndarray):
    """Compute RMSEs from x and y."""
    pred = (data_xy.x @ coefs_array).T
    return np.array([rmse(data_xy.y, p) for p in pred])


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


def compute_rmse_seq(data_xy: PolymlpDataXY, coefs_array: np.ndarray):
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
    return np.array(rmse_array)


def compute_rmse(data_xy: PolymlpDataXY, coefs_array: np.ndarray):
    """Compute RMSEs from xtx and xty."""
    if data_xy.xtx is None:
        return compute_rmse_standard(data_xy, coefs_array)
    return compute_rmse_seq(data_xy, coefs_array)


def get_best_model(
    coefs_array: np.ndarray,
    scales: np.ndarray,
    alphas: np.ndarray,
    rmse_train: np.ndarray,
    rmse_test: np.ndarray,
):
    """Return best polymlp model."""

    idx = np.argmin(rmse_test)
    best_model = PolymlpDataMLP(
        coeffs=coefs_array[:, idx],
        scales=scales,
        rmse_train=rmse_train[idx],
        rmse_test=rmse_test[idx],
        alpha=alphas[idx],
    )
    return best_model


def check_singular(rmse: np.ndarray, error_threshold: float = 1e6):
    """Check whether X.T @ X + penalty is ill-defined."""
    if np.all(np.array(rmse) > error_threshold):
        raise RuntimeError(
            "Matrix (X.T @ X + alpha * I) may be singular. "
            "This singularity issue might be reduced by increasing "
            "the value of alpha (the magnitude of the penalty term)."
        )


def print_log(
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
