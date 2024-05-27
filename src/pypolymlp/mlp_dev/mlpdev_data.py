#!/usr/bin/env python 
import numpy as np

from pypolymlp.mlp_dev.mlpdev_core import PolymlpDevParams
from pypolymlp.mlp_dev.mlpdev_data_base import PolymlpDevBase


class PolymlpDev(PolymlpDevBase):

    def __init__(self, params: PolymlpDevParams):
        """
        Keys in reg_dict
        ----------------
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        - scales
        """
        super().__init__(params)

    def run(self):

        self.compute_features()
        self.apply_scales()
        self.apply_weights()
        return self

    def compute_features(self):

        f_obj_train = self.features_class(self.params_dict, self.train_dict)
        f_obj_test = self.features_class(self.params_dict, self.test_dict)

        self.train_regression_dict = f_obj_train.regression_dict
        self.test_regression_dict = f_obj_test.regression_dict

        return self


class PolymlpDevSequential(PolymlpDevBase):

    def __init__(self, params: PolymlpDevParams):
        """
        Keys in reg_dict
        ----------------
        - x.T @ X
        - x.T @ y
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        - scales
        """
        super().__init__(params)

    def run(self):

        return self

    #def select_features_class(self):
    #    if self.is_hybrid == False:
    #        feature_func = Features

    def compute_features(self):

        if self.is_hybrid == False:
            f_obj_train = Features(self.params_dict, self.train_dict)

            f_obj_test = Features(self.params_dict, self.test_dict)
        else:
            f_obj_train = FeaturesHybrid(
                self.hybrid_params_dicts, self.train_dict
            )
            f_obj_test = FeaturesHybrid(
                self.hybrid_params_dicts, self.test_dict
            )

        self.__train_reg_dict = f_obj_train.regression_dict
        self.__test_reg_dict = f_obj_test.regression_dict

        return self


