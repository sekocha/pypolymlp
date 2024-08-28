"""Class for regression."""

from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataMLP, PolymlpDataXY
from pypolymlp.mlp_dev.core.regression_base import RegressionBase
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY


class Regression(RegressionBase):

    def __init__(
        self,
        polymlp_dev: PolymlpDevDataXY,
        train_xy: Optional[PolymlpDataXY] = None,
        test_xy: Optional[PolymlpDataXY] = None,
        verbose: bool = True,
    ):
        """Init method."""

        super().__init__(
            polymlp_dev,
            train_xy=train_xy,
            test_xy=test_xy,
            verbose=verbose,
        )
        self._alphas = [pow(10, a) for a in polymlp_dev.common_params.regression_alpha]
        self.polymlp_dev = polymlp_dev

    def fit(self, seq=False, clear_data=False, batch_size=128):
        """Estimate polymlp coefficients.

        Parameters
        ----------
        seq: Use sequential regression.
        clear_data: Clear training X.T @ X data to reduce memory
        batch_size: Batch size for computing test X.T @ X.
        """
        if seq:
            coefs_array = self._ridge_fit(A=self.train_xy.xtx, Xy=self.train_xy.xty)
            self._ridge_model_selection_seq(
                coefs_array, clear_data=clear_data, batch_size=batch_size
            )
        else:
            coefs_array = self._ridge_fit(X=self.train_xy.x, y=self.train_xy.y)
            self._ridge_model_selection(coefs_array)
        return self

    def _ridge_fit(self, X=None, y=None, A=None, Xy=None):

        A, Xy = self.compute_inner_products(X=X, y=y, A=A, Xy=Xy)
        n_features = A.shape[0]

        if self.verbose:
            print("Regression: cholesky decomposition ...", flush=True)
        coefs_array = np.zeros((n_features, len(self._alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(self._alphas):
            if self.verbose:
                print("- alpha:", alpha, flush=True)
            add = alpha - alpha_prev
            A.flat[:: n_features + 1] += add
            coefs_array[:, i] = self.solve_linear_equation(A, Xy)
            alpha_prev = alpha
        A.flat[:: n_features + 1] -= alpha
        return coefs_array

    def _ridge_model_selection(self, coefs_array):

        pred_train, pred_test, rmse_train, rmse_test = self.predict(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = PolymlpDataMLP(
            coeffs=coefs_array[:, idx],
            scales=self._scales,
            rmse=rmse_test[idx],
            alpha=self._alphas[idx],
            predictions_train=pred_train[idx],
            predictions_test=pred_test[idx],
        )
        if self.verbose:
            self._print_log(rmse_train, rmse_test)

        return self

    def _ridge_model_selection_seq(
        self,
        coefs_array,
        clear_data=False,
        batch_size=128,
    ):
        if clear_data:
            rmse_train = self.predict_seq_train(coefs_array)
            self._clear_train()
            self._calc_assign_test(batch_size=batch_size)
            rmse_test = self.predict_seq_test(coefs_array)
            self._clear_test()
        else:
            rmse_train, rmse_test = self.predict_seq(coefs_array)

        idx = np.argmin(rmse_test)
        self.best_model = PolymlpDataMLP(
            coeffs=coefs_array[:, idx],
            scales=self._scales,
            rmse=rmse_test[idx],
            alpha=self._alphas[idx],
        )
        if self.verbose:
            self._print_log(rmse_train, rmse_test)

        return self

    def _clear_train(self):
        if self.verbose:
            print("Clear training X.T @ X", flush=True)
        self.delete_train_xy()

    def _clear_test(self):
        if self.verbose:
            print("Clear test X.T @ X", flush=True)
        self.delete_test_xy()

    def _calc_assign_test(self, batch_size=128):
        if self.verbose:
            print("Calculate X.T @ X for test data", flush=True)
        self.polymlp_dev.run_test(element_swap=False, batch_size=batch_size)
        self.test_xy = self.polymlp_dev.test_xy

    def _print_log(self, rmse_train, rmse_test):
        print("Regression: model selection ...", flush=True)
        for a, rmse1, rmse2 in zip(self._alphas, rmse_train, rmse_test):
            if rmse1 > 1e6:
                print(
                    "- alpha =",
                    "{:.3e}".format(a),
                    ": rmse (train, test) =",
                    "{:.5e}".format(rmse1),
                    "{:.5e}".format(rmse2),
                    flush=True,
                )
            else:
                print(
                    "- alpha =",
                    "{:.3e}".format(a),
                    ": rmse (train, test) =",
                    "{:.5f}".format(rmse1),
                    "{:.5f}".format(rmse2),
                    flush=True,
                )
        return self
