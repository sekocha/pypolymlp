"""Class for regression using trasfer learning."""

import itertools
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataMLP, PolymlpDataXY
from pypolymlp.mlp_dev.core.regression_base import RegressionBase
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY


class RegressionTransfer(RegressionBase):
    """Class for regression using trasfer learning."""

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

        alphas = [pow(10, a) for a in polymlp_dev.common_params.regression_alpha]
        betas = [pow(10, a) for a in range(-5, 5)]
        self._params_reg = list(itertools.product(alphas, betas))
        self._coeffs_regular = None
        self.polymlp_dev = polymlp_dev

    def _set_coeffs_regular(self, coeffs, scales):

        if self.is_hybrid:
            coeffs_regular = np.array([c2 for c1 in coeffs for c2 in c1])
            scales_regular = np.array([c2 for c1 in scales for c2 in c1])
        else:
            coeffs_regular = coeffs
            scales_regular = scales

        """This part is better to be reconsidered."""
        coeffs_regular = coeffs_regular / scales_regular
        self._coeffs_regular = coeffs_regular * self._scales
        return self._coeffs_regular

    def fit(
        self,
        coeffs: np.ndarray,
        scales: np.ndarray,
        seq: bool = False,
        clear_data: bool = False,
        batch_size: int = 128,
    ):
        """Estimate polymlp coefficients.

        Parameters
        ----------
        coeffs: polymlp coefficients for domain model.
        scales: scales for domain model.
        seq: Use sequential regression.
        clear_data: Clear training X.T @ X data to reduce memory
        batch_size: Batch size for computing test X.T @ X.
        """

        self._coeffs_regular = self._set_coeffs_regular(coeffs, scales)
        if seq:
            coefs_array = self._regularization_fit(
                A=self.train_xy.xtx,
                Xy=self.train_xy.xty,
            )
            self._model_selection_seq(
                coefs_array,
                clear_data=clear_data,
                batch_size=batch_size,
            )
        else:
            coefs_array = self._regularization_fit(X=self.train_xy.x, y=self.train_xy.y)
            self._model_selection(coefs_array)

        return self

    def _regularization_fit(self, X=None, y=None, A=None, Xy=None):

        A, Xy = self.compute_inner_products(X=X, y=y, A=A, Xy=Xy)
        n_features = A.shape[0]

        if self.verbose:
            print("Regression: cholesky decomposition ...", flush=True)
        coefs_array = np.zeros((n_features, len(self._params_reg)))
        alpha_prev, beta_prev = 0.0, 0.0
        for i, (alpha, beta) in enumerate(self._params_reg):
            print(
                " (alpha, beta):",
                "{:.3e}".format(alpha),
                "{:.3e}".format(beta),
                flush=True,
            )
            add1 = (alpha + beta) - (alpha_prev + beta_prev)
            add2 = beta - beta_prev
            A.flat[:: n_features + 1] += add1
            Xy += add2 * self._coeffs_regular
            coefs_array[:, i] = self.solve_linear_equation(A, Xy)
            alpha_prev = alpha
            beta_prev = beta

        A.flat[:: n_features + 1] -= alpha + beta
        Xy -= beta * self._coeffs_regular
        return coefs_array

    def _model_selection(self, coefs_array, iprint=True):

        pred_train, pred_test, rmse_train, rmse_test = self.predict(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = PolymlpDataMLP(
            coeffs=coefs_array[:, idx],
            scales=self._scales,
            rmse=rmse_test[idx],
            alpha=self._params_reg[idx][0],
            beta=self._params_reg[idx][1],
            predictions_train=pred_train[idx],
            predictions_test=pred_test[idx],
        )
        if self.verbose:
            self._print_log(rmse_train, rmse_test)

        return self

    def _model_selection_seq(
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
            alpha=self._params_reg[idx][0],
            beta=self._params_reg[idx][1],
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
        for (a, b), rmse1, rmse2 in zip(self._params_reg, rmse_train, rmse_test):
            print(
                " - (alpha, beta) =",
                "{:.3e}".format(a),
                "{:.3e}".format(b),
                ": rmse (train, test) =",
                "{:.5f}".format(rmse1),
                "{:.5f}".format(rmse2),
                flush=True,
            )
