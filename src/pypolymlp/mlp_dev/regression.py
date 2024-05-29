#!/usr/bin/env python 
import numpy as np
from math import sqrt
from scipy.linalg.lapack import get_lapack_funcs

from pypolymlp.core.utils import rmse
from pypolymlp.core.io_polymlp import (
    save_mlp_lammps, save_multiple_mlp_lammps
)
from pypolymlp.mlp_dev.mlpdev_data import PolymlpDev


class Regression:

    def __init__(
        self, 
        polymlp_dev: PolymlpDev, 
        train_regression_dict=None,
        test_regression_dict=None,
    ):

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
        self.__best_model['scales'] = self.__scales = self.__vtrain['scales']
        self.__coeffs = None

    def ridge(self, iprint=True):

        alphas = [pow(10, a) for a in self.__common_params_dict['reg']['alpha']]
        coefs_array = self.__ridge_fit(
            X=self.__vtrain['x'], y=self.__vtrain['y'], alphas=alphas
        )
        self.__best_model = self.__ridge_model_selection(
            alphas, coefs_array, iprint=iprint
        )
        self.__coeffs = self.__best_model['coeffs']

        return self

    def ridge_seq(self, iprint=True):

        alphas = [pow(10, a) for a in self.__common_params_dict['reg']['alpha']]
        coefs_array = self.__ridge_fit(
            A=self.__vtrain['xtx'], Xy=self.__vtrain['xty'], alphas=alphas
        )
        self.__best_model = self.__ridge_model_selection_seq(
            alphas, coefs_array, iprint=iprint
        )
        self.__coeffs = self.__best_model['coeffs']

        return self

    def __ridge_fit(self, X=None, y=None, A=None, Xy=None, alphas=[1e-3,1e-1]):

        if X is not None and y is not None:
            print('  regression: computing inner products ...')
            n_samples, n_features = X.shape
            A = np.dot(X.T, X)
            Xy = np.dot(X.T, y)
        else:
            n_features = A.shape[0]

        print('  regression: cholesky decomposition ...')
        coefs_array = np.zeros((n_features, len(alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(alphas):
            add = alpha - alpha_prev
            A.flat[::n_features + 1] += add
            coefs_array[:,i] = self.__solve_linear_equation(A, Xy)
            alpha_prev = alpha
        A.flat[::n_features + 1] -= alpha

        return coefs_array

    def __solve_linear_equation(self, A, b):
        """
        numpy and scipy implementations
        x = np.linalg.solve(A, b)
        x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
        """
        posv, = get_lapack_funcs(('posv',), (A, b))
        _, x, _ = posv(A, b, lower=False,
                       overwrite_a=False,
                       overwrite_b=False)
        return x

    def __ridge_model_selection(self, alpha_array, coefs_array, iprint=True):

        pred_train_array = np.dot(self.__vtrain['x'], coefs_array).T
        pred_test_array = np.dot(self.__vtest['x'], coefs_array).T
        rmse_train_array = [
            rmse(self.__vtrain['y'], p) for p in pred_train_array
        ]
        rmse_test_array = [rmse(self.__vtest['y'], p) for p in pred_test_array]

        idx = np.argmin(rmse_test_array)
        self.__best_model['rmse'] = rmse_test_array[idx]
        self.__best_model['coeffs'] = coefs_array[:,idx]
        self.__best_model['alpha'] = alpha_array[idx]
        self.__best_model['predictions'] = dict()
        self.__best_model['predictions']['train'] = pred_train_array[idx]
        self.__best_model['predictions']['test'] = pred_test_array[idx]
 
        if iprint == True:
            print('  regression: model selection ...')
            for a, rmse1, rmse2 in zip(alpha_array, 
                                       rmse_train_array, 
                                       rmse_test_array):
                print('  - alpha =', '{:f}'.format(a), 
                      ': rmse (train, test) =', 
                      '{:f}'.format(rmse1), '{:f}'.format(rmse2))

        return self.__best_model

    def __ridge_model_selection_seq(self, 
                                    alpha_array, 
                                    coefs_array, 
                                    iprint=True):
        
        # computing rmse using xtx, xty and y_sq
        rmse_train_array, rmse_test_array = [], []
        for coefs in coefs_array.T:
            mse_train = self.__compute_mse(
                self.__vtrain['xtx'], self.__vtrain['xty'],
                self.__vtrain['y_sq_norm'], self.__vtrain['total_n_data'],
                coefs
            )
            mse_test = self.__compute_mse(
                self.__vtest['xtx'], self.__vtest['xty'],
                self.__vtest['y_sq_norm'], self.__vtest['total_n_data'],
                coefs
            )
            try:
                rmse_train_array.append(sqrt(mse_train))
            except:
                rmse_train_array.append(0.0)
            rmse_test_array.append(sqrt(mse_test))

        idx = np.argmin(rmse_test_array)
        self.__best_model['rmse'] = rmse_test_array[idx]
        self.__best_model['coeffs'] = coefs_array[:,idx]
        self.__best_model['alpha'] = alpha_array[idx]
  
        if iprint == True:
            print('  regression: model selection ...')
            for a, rmse1, rmse2 in zip(alpha_array, 
                                       rmse_train_array, 
                                       rmse_test_array):
                print('  - alpha =', '{:f}'.format(a), 
                      ': rmse (train, test) =', 
                      '{:f}'.format(rmse1), '{:f}'.format(rmse2))

        return self.__best_model

    def __compute_mse(self, xtx, xty, y_sq_norm, size, coefs):

        v1 = np.dot(coefs, np.dot(xtx, coefs))
        v2 = - 2 * np.dot(coefs, xty)
        return (v1 + v2 + y_sq_norm) / size

    def save_mlp_lammps(self, filename='polymlp.lammps'):

        if self.__hybrid == False:
            save_mlp_lammps(
                self.__params_dict, 
                self.__coeffs, 
                self.__scales, 
                filename=filename
            )
        else:
            save_multiple_mlp_lammps(
                self.__params_dict,
                self.__vtrain['cumulative_n_features'],
                self.__coeffs,
                self.__scales,
            )
        return self

    def hybrid_division(self, target):

        cumulative = self.__vtrain['cumulative_n_features']
        list_target = []
        for i, params_dict in enumerate(self.__params_dict):
            if i == 0:
                begin, end = 0, cumulative[0]
            else:
                begin, end = cumulative[i-1], cumulative[i]
            list_target.append(target[begin:end])
        return list_target

    @property
    def best_model(self):
        """
        Keys
        ----
        coeffs, scales, rmse, alpha, predictions (train, test)
        """
        self.__best_model['coeffs'] = self.coeffs
        self.__best_model['scales'] = self.scales
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

    @coeffs.setter
    def coeffs(self, array):
        self.__coeffs = array

    @scales.setter
    def scales(self, array):
        self.__scales = array

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
    def is_multiple_datasets(self):
        return self.__multiple_datasets

    @property
    def is_hybrid(self):
        return self.__hybrid


