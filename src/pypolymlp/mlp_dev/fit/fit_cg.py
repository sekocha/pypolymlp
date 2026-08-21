"""Class for estimating MLP coefficients using conjugate gradient."""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore

from .fit_base import PolymlpFitBase
from .solvers_cg import solver_cg


def _check_use_xy(polymlp: PolymlpDevCore):
    """Check whether xtx and xty data is used or not."""
    try:
        polymlp.check_memory_size_in_regression()
    except RuntimeError:
        return True
    return False


class PolymlpFitCG(PolymlpFitBase):
    """Class for estimating MLP coefficients using conjugate gradient."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        test: DatasetList,
        gtol: float = 1e-2,
        max_iter: Optional[int] = None,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train: Training datasets.
        test: Test datasets.
        """
        # TODO: Activate batch size.
        super().__init__(params, train, use_gradient=True, verbose=verbose)

        self._test = test
        self._gtol = gtol
        self._max_iter = max_iter

    def fit(self):
        """Estimate MLP coefficients."""
        use_xy = _check_use_xy(self._polymlp)
        calc_features = self._polymlp.calc_xy if use_xy else self._polymlp.calc_xtx_xty
        if self._verbose:
            print("Use X.T @ X:", np.logical_not(use_xy), flush=True)

        train_xy = calc_features(self._train)
        if self._max_iter is None:
            self._max_iter = max(self._polymlp.n_features * 3, 50000)

        coefs, coef0 = [], None
        for alpha in reversed(self._params.alphas):
            c = solver_cg(
                x=train_xy.x,
                y=train_xy.y,
                xtx=train_xy.xtx,
                xty=train_xy.xty,
                alpha=alpha,
                coef0=coef0,
                gtol=self._gtol,
                max_iter=self._max_iter,
                verbose=self._verbose,
            )
            coef0 = c
            coefs.append(c)
        coefs = np.array(coefs)[::-1].T

        rmse_train = self._polymlp.compute_rmse(coefs, train_xy, check_singular=True)
        train_xy.clear_data()

        test_xy = calc_features(
            self._test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
        )
        rmse_test = self._polymlp.compute_rmse(coefs, test_xy)
        test_xy.clear_data()

        self._best_model = self._polymlp.get_best_model(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        self._all_models = self._polymlp.get_all_models(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        if self._verbose:
            self._polymlp.print_model_selection_log(rmse_train, rmse_test)

        return self
