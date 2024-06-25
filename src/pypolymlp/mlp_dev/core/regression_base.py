#!/usr/bin/env python
from abc import ABC, abstractmethod
from math import sqrt

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from pypolymlp.core.io_polymlp import save_mlp_lammps, save_multiple_mlp_lammps
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase


class RegressionBase(ABC):

    def __init__(
        self,
        polymlp_dev: PolymlpDevDataXYBase,
        train_regression_dict=None,
        test_regression_dict=None,
        verbose=True,
    ):

        self.__verbose = verbose
        self.__params_dict = polymlp_dev.params_dict
        self.__common_params_dict = polymlp_dev.common_params_dict
        self.__hybrid = polymlp_dev.is_hybrid

        self.__multiple_datasets = polymlp_dev.is_multiple_datasets

        if train_regression_dict is None:
            self.__vtrain = polymlp_dev.train_regression_dict
        else:
            self.__vtrain = train_regression_dict

        if test_regression_dict is None:
            self.__vtest = polymlp_dev.test_regression_dict
        else:
            self.__vtest = test_regression_dict

        self.__train_dict = polymlp_dev.train_dict
        self.__test_dict = polymlp_dev.test_dict

        self.__best_model = dict()
        self.__best_model["scales"] = self.__scales = self.__vtrain["scales"]
        self.__coeffs = None

    @abstractmethod
    def fit(self):
        pass

    def solve_linear_equation(self, A, b):
        """
        numpy and scipy implementations
        x = np.linalg.solve(A, b)
        x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
        """
        (posv,) = get_lapack_funcs(("posv",), (A, b))
        _, x, _ = posv(A, b, lower=False, overwrite_a=False, overwrite_b=False)
        return x

    def compute_inner_products(self, X=None, y=None, A=None, Xy=None):

        if X is not None and y is not None:
            if self.__verbose:
                print("Regression: computing inner products ...")
            A = np.dot(X.T, X)
            Xy = np.dot(X.T, y)
        return A, Xy

    def rmse(self, true, pred):
        return rmse(true, pred)

    def rmse_list(self, true, pred_list):
        return [rmse(true, p) for p in pred_list]

    def predict(self, coefs_array):
        """computing rmse using X and y"""
        pred_train = np.dot(self.__vtrain["x"], coefs_array).T
        pred_test = np.dot(self.__vtest["x"], coefs_array).T
        rmse_train = self.rmse_list(self.__vtrain["y"], pred_train)
        rmse_test = self.rmse_list(self.__vtest["y"], pred_test)
        return pred_train, pred_test, rmse_train, rmse_test

    def predict_seq(self, coefs_array):
        """computing rmse using xtx, xty and y_sq"""
        rmse_train_array, rmse_test_array = [], []
        for coefs in coefs_array.T:
            mse_train = self.__compute_mse(
                self.__vtrain["xtx"],
                self.__vtrain["xty"],
                self.__vtrain["y_sq_norm"],
                self.__vtrain["total_n_data"],
                coefs,
            )
            mse_test = self.__compute_mse(
                self.__vtest["xtx"],
                self.__vtest["xty"],
                self.__vtest["y_sq_norm"],
                self.__vtest["total_n_data"],
                coefs,
            )
            try:
                rmse_train_array.append(sqrt(mse_train))
            except:
                rmse_train_array.append(0.0)

            rmse_test_array.append(sqrt(mse_test))

        return rmse_train_array, rmse_test_array

    def __compute_mse(self, xtx, xty, y_sq_norm, size, coefs):

        v1 = np.dot(coefs, np.dot(xtx, coefs))
        v2 = -2 * np.dot(coefs, xty)
        return (v1 + v2 + y_sq_norm) / size

    def save_mlp_lammps(self, filename="polymlp.lammps"):

        if self.__hybrid is False:
            save_mlp_lammps(
                self.__params_dict,
                self.__coeffs,
                self.__scales,
                filename=filename,
            )
        else:
            save_multiple_mlp_lammps(
                self.__params_dict,
                self.__vtrain["cumulative_n_features"],
                self.__coeffs,
                self.__scales,
            )
        return self

    def hybrid_division(self, target):

        cumulative = self.__vtrain["cumulative_n_features"]
        list_target = []
        for i, params_dict in enumerate(self.__params_dict):
            if i == 0:
                begin, end = 0, cumulative[0]
            else:
                begin, end = cumulative[i - 1], cumulative[i]
            list_target.append(np.array(target[begin:end]))
        return list_target

    @property
    def best_model(self):
        """
        Keys
        ----
        coeffs, scales, rmse, alpha, predictions (train, test)
        """
        return self.__best_model

    @property
    def coeffs(self):
        if self.__hybrid:
            return self.hybrid_division(self.__coeffs)
        return self.__coeffs

    @property
    def scales(self):
        if self.__hybrid:
            return self.hybrid_division(self.__scales)
        return self.__scales

    @best_model.setter
    def best_model(self, dict1):
        self.__best_model.update(dict1)
        self.__coeffs = self.__best_model["coeffs"]
        self.__scales = self.__best_model["scales"]

    @property
    def coeffs_vector(self):
        return self.__coeffs

    @property
    def scales_vector(self):
        return self.__scales

    @coeffs.setter
    def coeffs(self, array):
        self.__coeffs = array
        self.__best_model["coeffs"] = array

    @scales.setter
    def scales(self, array):
        self.__scales = array
        self.__best_model["scales"] = array

    @property
    def params_dict(self):
        return self.__params_dict

    @property
    def train_dict(self):
        return self.__train_dict

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def train_regression_dict(self):
        return self.__vtrain

    @property
    def test_regression_dict(self):
        return self.__vtest

    @property
    def is_multiple_datasets(self):
        return self.__multiple_datasets

    @property
    def is_hybrid(self):
        return self.__hybrid

    @property
    def verbose(self):
        return self.__verbose
