"""Functions for model selection."""

import numpy as np

from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.data import PolymlpDataXY


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


#        idx = np.argmin(rmse_test)
#        self.best_model = PolymlpDataMLP(
#            coeffs=coefs_array[:, idx],
#            scales=self._scales,
#            rmse=rmse_test[idx],
#            alpha=self._alphas[idx],
#        )

# idx = np.argmin(rmse_test)
#         self.best_model = PolymlpDataMLP(
#             coeffs=coefs_array[:, idx],
#             scales=self._scales,
#             rmse=rmse_test[idx],
#             alpha=self._alphas[idx],
#             predictions_train=pred_train[idx],
#             predictions_test=pred_test[idx],
#         )
#
