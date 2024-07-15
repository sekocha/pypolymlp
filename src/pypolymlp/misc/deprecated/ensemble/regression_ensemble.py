#!/usr/bin/env python
import numpy as np

from pypolymlp.mlp_dev.core.regression_base import RegressionBase
from pypolymlp.mlp_dev.ensemble.mlpdev_dataxy_ensemble_base import (
    PolymlpDevDataXYEnsembleBase,
)
from pypolymlp.mlp_dev.standard.regression import Regression


class RegressionEnsemble(RegressionBase):

    def __init__(
        self,
        polymlp: PolymlpDevDataXYEnsembleBase,
        train_regression_dict=None,
        test_regression_dict=None,
    ):
        super().__init__(
            polymlp,
            train_regression_dict=train_regression_dict,
            test_regression_dict=test_regression_dict,
        )

        self.__polymlp = polymlp

        self.__train_reg_dict_all = polymlp.train_regression_dict_list
        self.__test_reg_dict_all = polymlp.test_regression_dict_list
        self.__random_indices = polymlp.random_indices
        self.__n_features = polymlp.n_features
        self.__n_models = polymlp.n_models
        # self.__cumulative = polymlp.cumulative_n_features

    def fit(self):

        coeffs_sum = np.zeros(self.__n_features)
        for train_reg, test_reg, r_indices in zip(
            self.__train_reg_dict_all,
            self.__test_reg_dict_all,
            self.__random_indices,
        ):

            reg = Regression(
                self.__polymlp,
                train_regression_dict=train_reg,
                test_regression_dict=test_reg,
            )
            reg.fit(seq=True)
            coeffs_sum[r_indices] += reg.coeffs_vector / reg.scales_vector

            self.__train_dict = reg.train_dict
            self.__test_dict = reg.test_dict

        self.coeffs = coeffs_sum / self.__n_models
        self.scales = np.ones(self.__n_features)
        return self
