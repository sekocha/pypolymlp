#!/usr/bin/env python 
import numpy as np

from pypolymlp.core.io_polymlp import (
    save_mlp_lammps, save_multiple_mlp_lammps
)
from pypolymlp.mlp_dev.regression import Regression

from pypolymlp.mlp_dev.ensemble.mlpdev_data_feature_bagging import (
    PolymlpDevFeatureBagging,
    PolymlpDevFeatureBaggingSequential,
)


class RegressionEnsemble:
    
    def __init__(self, polymlp):

        self.__polymlp = polymlp
        self.__params_dict = polymlp.params_dict
        self.__common_params_dict = polymlp.common_params_dict

        self.__train_reg_dict_all = polymlp.train_regression_dict_list
        self.__test_reg_dict_all = polymlp.test_regression_dict_list
        self.__random_indices = polymlp.random_indices
        self.__n_features = polymlp.n_features
        self.__n_models = polymlp.n_models
        self.__hybrid = polymlp.is_hybrid
        self.__multiple_datasets = polymlp.is_multiple_datasets
        self.__cumulative = polymlp.cumulative_n_features

        self.__train_dict = None
        self.__test_dict = None

        self.__best_model = dict()
        self.__coeffs = None
        self.__scales = None

    def fit(self):

        coeffs_sum = np.zeros(self.__n_features)
        for train_reg, test_reg, r_indices in zip(
            self.__train_reg_dict_all, 
            self.__test_reg_dict_all, 
            self.__random_indices
        ):

            reg = Regression(
                self.__polymlp, 
                train_regression_dict=train_reg, 
                test_regression_dict=test_reg,
            )
            reg.ridge_seq()
            mlp_dict = reg.best_model
            coeffs_sum[r_indices] += mlp_dict['coeffs'] / mlp_dict['scales']

            self.__train_dict = reg.train_dict
            self.__test_dict = reg.test_dict

        self.__coeffs = coeffs_sum / self.__n_models
        self.__scales = np.ones(self.__n_features)
        return self

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
                self.__cumulative,
                self.__coeffs,
                self.__scales,
            )
        return self

    def hybrid_division(self, target):

        list_target = []
        for i, params_dict in enumerate(self.__params_dict):
            if i == 0:
                begin, end = 0, self.__cumulative[0]
            else:
                begin, end = self.__cumulative[i-1], self.__cumulative[i]
            list_target.append(target[begin:end])
        return list_target

    @property
    def best_model(self):
        """
        Keys
        ----
        coeffs, scales
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


