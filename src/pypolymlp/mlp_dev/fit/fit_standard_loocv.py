"""Classes for estimating regression coefficients from datasets."""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams

from .fit_base import PolymlpFitBase


class PolymlpFitStandardLOOCV(PolymlpFitBase):
    """Class for estimating MLP coefficients without computing entire X."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train: Training datasets.
        test: Test datasets.
        batch_size: Batch size for sequential regression.
                    If None, the batch size is automatically determined
                    depending on the memory size and number of features.
        """
        super().__init__(params, train, verbose=verbose)

        self._batch_size = batch_size
        self._inv_xtx = None

    def fit(self):
        """Estimate MLP coefficients."""
        self._polymlp.check_memory_size_in_regression()

        train_xy = self._polymlp.calc_xtx_xty(self._train, batch_size=self._batch_size)

        xtx = train_xy.xtx
        xty = train_xy.xty
        alphas = self._params.alphas
        if self._verbose:
            print("Regression:", flush=True)

        cv_scores = []
        n_features = xtx.shape[0]
        coefs_array = np.zeros((n_features, len(alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(alphas):
            if self._verbose:
                print("- alpha:", alpha, flush=True)
            add = alpha - alpha_prev
            if self._verbose:
                print("  Compute X.T @ X + alpha @ I", flush=True)
            xtx.flat[:: n_features + 1] += add
            if self._verbose:
                print("  Compute inverse matrix", flush=True)
            inv_xtx = np.linalg.inv(xtx)
            coefs_single = inv_xtx @ xty
            coefs_array[:, i] = coefs_single

            rmse_cv = self._polymlp.compute_rmse_cv(
                coefs_single,
                train_xy.scales,
                self._train,
                inv_xtx,
                batch_size=20,
            )
            cv_scores.append(rmse_cv)
            alpha_prev = alpha

        if self._verbose:
            self._polymlp.print_model_selection_log([0] * len(cv_scores), cv_scores)

        self._best_model = self._polymlp.get_best_model(
            coefs_array,
            train_xy.scales,
            [0] * len(cv_scores),
            cv_scores,
            train_xy.cumulative_n_features,
        )
        self._all_models = self._polymlp.get_all_models(
            coefs_array,
            train_xy.scales,
            [0] * len(cv_scores),
            cv_scores,
            train_xy.cumulative_n_features,
        )
        xtx.flat[:: n_features + 1] -= alpha
        xtx.flat[:: n_features + 1] += self._best_model.alpha
        self._inv_xtx = np.linalg.inv(xtx)
        return self

    @property
    def inv_xtx(self):
        """Return inverse of X.T @ X."""
        return self._inv_xtx
