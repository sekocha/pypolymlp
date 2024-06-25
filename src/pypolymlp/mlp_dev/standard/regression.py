#!/usr/bin/env python
import numpy as np

from pypolymlp.mlp_dev.core.regression_base import RegressionBase
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY


class Regression(RegressionBase):

    def __init__(
        self,
        polymlp_dev: PolymlpDevDataXY,
        train_regression_dict=None,
        test_regression_dict=None,
        verbose=True,
    ):

        super().__init__(
            polymlp_dev,
            train_regression_dict=train_regression_dict,
            test_regression_dict=test_regression_dict,
            verbose=verbose,
        )
        self.__alphas = [
            pow(10, a) for a in polymlp_dev.common_params_dict["reg"]["alpha"]
        ]

    def fit(self, seq=False):

        vtrain = self.train_regression_dict
        if seq:
            XTX, XTy = vtrain["xtx"], vtrain["xty"]
            coefs_array = self.__ridge_fit(A=XTX, Xy=XTy)
            self.__ridge_model_selection_seq(coefs_array)
        else:
            X, y = vtrain["x"], vtrain["y"]
            coefs_array = self.__ridge_fit(X=X, y=y)
            self.__ridge_model_selection(coefs_array)

        return self

    def __ridge_fit(self, X=None, y=None, A=None, Xy=None):

        A, Xy = self.compute_inner_products(X=X, y=y, A=A, Xy=Xy)
        n_features = A.shape[0]

        if self.verbose:
            print("Regression: cholesky decomposition ...")
        coefs_array = np.zeros((n_features, len(self.__alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(self.__alphas):
            add = alpha - alpha_prev
            A.flat[:: n_features + 1] += add
            coefs_array[:, i] = self.solve_linear_equation(A, Xy)
            alpha_prev = alpha
        A.flat[:: n_features + 1] -= alpha

        return coefs_array

    def __ridge_model_selection(self, coefs_array):

        pred_train, pred_test, rmse_train, rmse_test = self.predict(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = {
            "rmse": rmse_test[idx],
            "coeffs": coefs_array[:, idx],
            "alpha": self.__alphas[idx],
            "predictions": {
                "train": pred_train[idx],
                "test": pred_test[idx],
            },
        }
        if self.verbose:
            self.__print_log(rmse_train, rmse_test)

        return self

    def __ridge_model_selection_seq(self, coefs_array):

        rmse_train, rmse_test = self.predict_seq(coefs_array)
        idx = np.argmin(rmse_test)
        self.best_model = {
            "rmse": rmse_test[idx],
            "coeffs": coefs_array[:, idx],
            "alpha": self.__alphas[idx],
        }
        if self.verbose:
            self.__print_log(rmse_train, rmse_test)

        return self

    def __print_log(self, rmse_train, rmse_test):

        print("Regression: model selection ...")
        for a, rmse1, rmse2 in zip(self.__alphas, rmse_train, rmse_test):
            print(
                "  - alpha =",
                "{:.3e}".format(a),
                ": rmse (train, test) =",
                "{:.5f}".format(rmse1),
                "{:.5f}".format(rmse2),
            )
        return self
