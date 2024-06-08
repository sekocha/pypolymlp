#!/usr/bin/env python
import itertools

import numpy as np

from pypolymlp.mlp_dev.core.regression_base import RegressionBase
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY


class RegressionTransfer(RegressionBase):

    def __init__(
        self,
        polymlp_dev: PolymlpDevDataXY,
        train_regression_dict=None,
        test_regression_dict=None,
    ):

        super().__init__(
            polymlp_dev,
            train_regression_dict=train_regression_dict,
            test_regression_dict=test_regression_dict,
        )

        alphas = [pow(10, a) for a in polymlp_dev.common_params_dict["reg"]["alpha"]]
        betas = [pow(10, a) for a in range(-5, 5)]
        self.__params = list(itertools.product(alphas, betas))
        self.__coeffs_regular = None

    def __set_coeffs_regular(self, coeffs, scales):

        if self.is_hybrid:
            coeffs_regular = np.array([c2 for c1 in coeffs for c2 in c1])
            scales_regular = np.array([c2 for c1 in scales for c2 in c1])
        else:
            coeffs_regular = coeffs
            scales_regular = scales

        coeffs_regular = coeffs_regular / scales_regular
        self.__coeffs_regular = coeffs_regular * self.scales
        return self.__coeffs_regular

    def fit(self, coeffs, scales, seq=False, iprint=True):

        self.__coeffs_regular = self.__set_coeffs_regular(coeffs, scales)

        vtrain = self.train_regression_dict
        if seq is False:
            X, y = vtrain["x"], vtrain["y"]
            coefs_array = self.__regularization_fit(X=X, y=y)
            self.__model_selection(coefs_array, iprint=iprint)
        else:
            XTX, XTy = vtrain["xtx"], vtrain["xty"]
            coefs_array = self.__regularization_fit(A=XTX, Xy=XTy)
            self.__model_selection_seq(coefs_array, iprint=iprint)

        return self

    def __regularization_fit(self, X=None, y=None, A=None, Xy=None):

        A, Xy = self.compute_inner_products(X=X, y=y, A=A, Xy=Xy)
        n_features = A.shape[0]

        print("Regression: cholesky decomposition ...")
        coefs_array = np.zeros((n_features, len(self.__params)))
        alpha_prev, beta_prev = 0.0, 0.0
        for i, (alpha, beta) in enumerate(self.__params):
            print(
                " (alpha, beta):",
                "{:.3e}".format(alpha),
                "{:.3e}".format(beta),
            )
            add1 = (alpha + beta) - (alpha_prev + beta_prev)
            add2 = beta - beta_prev
            A.flat[:: n_features + 1] += add1
            Xy += add2 * self.__coeffs_regular

            coefs_array[:, i] = self.solve_linear_equation(A, Xy)
            alpha_prev = alpha
            beta_prev = beta

        A.flat[:: n_features + 1] -= alpha + beta
        Xy -= beta * self.__coeffs_regular

        return coefs_array

    def __model_selection(self, coefs_array, iprint=True):

        pred_train, pred_test, rmse_train, rmse_test = self.predict(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = {
            "rmse": rmse_test[idx],
            "coeffs": coefs_array[:, idx],
            "alpha": self.__params[idx][0],
            "beta": self.__params[idx][1],
            "predictions": {
                "train": pred_train[idx],
                "test": pred_test[idx],
            },
        }
        if iprint:
            self.__print_log(rmse_train, rmse_test)

        return self

    def __model_selection_seq(self, coefs_array, iprint=True):

        rmse_train, rmse_test = self.predict_seq(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = {
            "rmse": rmse_test[idx],
            "coeffs": coefs_array[:, idx],
            "alpha": self.__params[idx][0],
            "beta": self.__params[idx][1],
        }
        if iprint:
            self.__print_log(rmse_train, rmse_test)

        return self

    def __print_log(self, rmse_train, rmse_test):

        print("Regression: model selection ...")
        for (a, b), rmse1, rmse2 in zip(self.__params, rmse_train, rmse_test):
            print(
                " - (alpha, beta) =",
                "{:.3e}".format(a),
                "{:.3e}".format(b),
                ": rmse (train, test) =",
                "{:.5f}".format(rmse1),
                "{:.5f}".format(rmse2),
            )
