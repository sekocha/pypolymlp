#!/usr/bin/env python
from math import sqrt

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from pypolymlp.core.utils import rmse


class Regression:

    def __init__(self, train_reg_dict, test_reg_dict, params_dict):

        self.params_dict = params_dict
        self.vtrain = train_reg_dict
        self.vtest = test_reg_dict

        self.best_model = dict()
        self.best_model["scales"] = self.scales = train_reg_dict["scales"]

    def get_best_model(self):
        """
        best_model:
            keys: coeffs, rmse, alpha, predictions (train, test)
        """
        return self.best_model

    def test_pca_projection(self, iprint=True, k=8000):

        from sklearn.decomposition import PCA

        alphas = [pow(10, a) for a in self.params_dict["reg"]["alpha"]]

        n_data = self.vtrain["x"].shape[0]
        n_samples = min(2 * k, n_data)
        samples = np.random.choice(range(n_data), size=n_samples, replace=False)

        print("Running PCA")
        pca = PCA(n_components=k).fit(self.vtrain["x"][samples])
        X = self.vtrain["x"] @ pca.components_.T
        print("Finished.")

        coefs_array = self.__ridge_fit(X=X, y=self.vtrain["y"], alphas=alphas)
        coefs_array = pca.components_.T @ coefs_array

        best_model = self.__ridge_model_selection(alphas, coefs_array, iprint=iprint)

        return best_model["coeffs"], self.scales

    def test_random_projection(self, iprint=True, k=3000):

        n_features = self.vtrain["x"].shape[1]
        alphas = [pow(10, a) for a in self.params_dict["reg"]["alpha"]]

        #        #element_list = [1, -1, 0]
        #        #prob_list = [1/6, 1/6, 2/3]
        #        element_list = [1, -1]
        #        prob_list = [1/2, 1/2]
        #
        #        random_proj = np.random.choice(
        #            a=element_list, size=(n_features, k), p=prob_list
        #        )

        nonzero = np.random.choice(range(n_features), size=k, replace=False)
        random_proj = np.zeros((n_features, k))
        for i, row in enumerate(nonzero):
            random_proj[row][i] = 1

        #        random_proj = np.random.uniform(-1.0, 1.0, (n_features, k))
        #        random_proj = random_proj / np.linalg.norm(random_proj, axis=0)
        """More efficient algorithm can be applied."""
        X = self.vtrain["x"] @ random_proj

        coefs_array = self.__ridge_fit(X=X, y=self.vtrain["y"], alphas=alphas)

        """Reconstruct coefs using random_proj, random_proj @ coefs_array"""
        coefs_array = random_proj @ coefs_array
        best_model = self.__ridge_model_selection(alphas, coefs_array, iprint=iprint)

        return best_model["coeffs"], self.scales

    def ridge(self, iprint=True):

        alphas = [pow(10, a) for a in self.params_dict["reg"]["alpha"]]
        coefs_array = self.__ridge_fit(
            X=self.vtrain["x"], y=self.vtrain["y"], alphas=alphas
        )
        best_model = self.__ridge_model_selection(alphas, coefs_array, iprint=iprint)

        return best_model["coeffs"], self.scales

    def ridge_seq(self, iprint=True):

        alphas = [pow(10, a) for a in self.params_dict["reg"]["alpha"]]
        coefs_array = self.__ridge_fit(
            A=self.vtrain["xtx"], Xy=self.vtrain["xty"], alphas=alphas
        )
        best_model = self.__ridge_model_selection_seq(
            alphas, coefs_array, iprint=iprint
        )
        return best_model["coeffs"], self.scales

    def __ridge_fit(self, X=None, y=None, A=None, Xy=None, alphas=[1e-3, 1e-1]):

        if X is not None and y is not None:
            print("  regression: computing inner products ...")
            n_samples, n_features = X.shape
            A = np.dot(X.T, X)
            Xy = np.dot(X.T, y)
        else:
            n_features = A.shape[0]

        print("  regression: cholesky decomposition ...")
        coefs_array = np.zeros((n_features, len(alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(alphas):
            add = alpha - alpha_prev
            A.flat[:: n_features + 1] += add
            coefs_array[:, i] = self.__solve_linear_equation(A, Xy)
            alpha_prev = alpha
        A.flat[:: n_features + 1] -= alpha

        return coefs_array

    def __solve_linear_equation(self, A, b):
        """
        numpy and scipy implementations
        x = np.linalg.solve(A, b)
        x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
        """
        (posv,) = get_lapack_funcs(("posv",), (A, b))
        _, x, _ = posv(A, b, lower=False, overwrite_a=False, overwrite_b=False)
        return x

    def __ridge_model_selection(self, alpha_array, coefs_array, iprint=True):

        pred_train_array = np.dot(self.vtrain["x"], coefs_array).T
        pred_test_array = np.dot(self.vtest["x"], coefs_array).T
        rmse_train_array = [rmse(self.vtrain["y"], p) for p in pred_train_array]
        rmse_test_array = [rmse(self.vtest["y"], p) for p in pred_test_array]

        idx = np.argmin(rmse_test_array)
        self.best_model["rmse"] = rmse_test_array[idx]
        self.best_model["coeffs"] = coefs_array[:, idx]
        self.best_model["alpha"] = alpha_array[idx]
        self.best_model["predictions"] = dict()
        self.best_model["predictions"]["train"] = pred_train_array[idx]
        self.best_model["predictions"]["test"] = pred_test_array[idx]

        if iprint == True:
            print("  regression: model selection ...")
            for a, rmse1, rmse2 in zip(alpha_array, rmse_train_array, rmse_test_array):
                print(
                    "  - alpha =",
                    "{:f}".format(a),
                    ": rmse (train, test) =",
                    "{:f}".format(rmse1),
                    "{:f}".format(rmse2),
                )

        return self.best_model

    def __ridge_model_selection_seq(self, alpha_array, coefs_array, iprint=True):

        # computing rmse using xtx, xty and y_sq
        rmse_train_array, rmse_test_array = [], []
        for coefs in coefs_array.T:
            mse_train = self.__compute_mse(
                self.vtrain["xtx"],
                self.vtrain["xty"],
                self.vtrain["y_sq_norm"],
                self.vtrain["total_n_data"],
                coefs,
            )
            mse_test = self.__compute_mse(
                self.vtest["xtx"],
                self.vtest["xty"],
                self.vtest["y_sq_norm"],
                self.vtest["total_n_data"],
                coefs,
            )
            rmse_train_array.append(sqrt(mse_train))
            rmse_test_array.append(sqrt(mse_test))

        idx = np.argmin(rmse_test_array)
        self.best_model["rmse"] = rmse_test_array[idx]
        self.best_model["coeffs"] = coefs_array[:, idx]
        self.best_model["alpha"] = alpha_array[idx]

        if iprint == True:
            print("  regression: model selection ...")
            for a, rmse1, rmse2 in zip(alpha_array, rmse_train_array, rmse_test_array):
                print(
                    "  - alpha =",
                    "{:f}".format(a),
                    ": rmse (train, test) =",
                    "{:f}".format(rmse1),
                    "{:f}".format(rmse2),
                )

        return self.best_model

    def __compute_mse(self, xtx, xty, y_sq_norm, size, coefs):
        v1 = np.dot(coefs, np.dot(xtx, coefs))
        v2 = -2 * np.dot(coefs, xty)
        return (v1 + v2 + y_sq_norm) / size

    def lasso(self, iprint=True):

        from sklearn.linear_model import LassoLars

        alphas = [pow(10, a) for a in self.params_dict["reg"]["alpha"]]

        best_rmse = 1e10
        for alpha in alphas:
            reg = LassoLars(alpha=alpha, fit_intercept=False)
            reg.fit(self.vtrain["x"], self.vtrain["y"])
            coeffs = reg.coef_
            pred_train = np.dot(self.vtrain["x"], coeffs)
            pred_test = np.dot(self.vtest["x"], coeffs)
            rmse_train = rmse(self.vtrain["y"], pred_train)
            rmse_test = rmse(self.vtest["y"], pred_test)
            if rmse_test < best_rmse:
                self.best_model["rmse"] = rmse_test
                self.best_model["coeffs"] = coeffs
                self.best_model["alpha"] = alpha
                self.best_model["predictions"] = dict()
                self.best_model["predictions"]["train"] = pred_train
                self.best_model["predictions"]["test"] = pred_test

            if iprint == True:
                print(
                    "  - alpha =",
                    "{:f}".format(a),
                    ": rmse (train, test) =",
                    "{:f}".format(rmse_train),
                    "{:f}".format(rmse_test),
                )

        return best_model["coeffs"], self.scales
