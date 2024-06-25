#!/usr/bin/env python
from abc import ABC, abstractmethod

import numpy as np

from pypolymlp.mlp_dev.core.features import Features, FeaturesHybrid
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.utils_weights import apply_weight_percentage


class PolymlpDevDataXYBase(ABC):

    def __init__(self, params: PolymlpDevData, verbose=True):
        """
        Keys in reg_dict
        ----------------
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        - scales
        """
        self.__params_dict = params.params_dict
        self.__common_params_dict = params.common_params_dict
        self.__verbose = verbose

        self.__train_dict = params.train_dict
        self.__test_dict = params.test_dict
        self.__min_energy = params.min_energy

        self.__multiple_datasets = params.is_multiple_datasets
        self.__hybrid = params.is_hybrid

        if self.is_hybrid is False:
            self.__feature_class = Features
        else:
            self.__feature_class = FeaturesHybrid

        self.__train_reg_dict = None
        self.__test_reg_dict = None
        self.__scales = None

    def print_data_shape(self):

        x = self.__train_reg_dict["x"]
        ne, nf, ns = self.__train_reg_dict["n_data"]
        print("Training Dataset:", x.shape)
        print("   - n (energy) =", ne)
        print("   - n (force)  =", nf)
        print("   - n (stress) =", ns)

        x = self.__test_reg_dict["x"]
        ne, nf, ns = self.__test_reg_dict["n_data"]
        print("Test Dataset:", x.shape)
        print("   - n (energy) =", ne)
        print("   - n (force)  =", nf)
        print("   - n (stress) =", ns)
        return self

    def apply_scales(self):

        if self.__train_reg_dict is None:
            raise ValueError("Not found: regression_dict.")

        x = self.__train_reg_dict["x"]
        ne, nf, ns = self.__train_reg_dict["n_data"]
        self.__scales = np.std(x[:ne], axis=0)

        self.__train_reg_dict["x"] /= self.__scales
        self.__test_reg_dict["x"] /= self.__scales

        self.__train_reg_dict["scales"] = self.__scales
        self.__test_reg_dict["scales"] = self.__scales

        return self

    def apply_weights(self, weight_stress=0.1):

        if self.__train_reg_dict is None:
            raise ValueError("Not found: regression_dict.")

        self.__train_reg_dict = self.__apply_weights_single_set(
            self.__train_dict,
            self.__train_reg_dict,
            weight_stress=weight_stress,
        )
        self.__test_reg_dict = self.__apply_weights_single_set(
            self.__test_dict,
            self.__test_reg_dict,
            weight_stress=weight_stress,
        )

    def __apply_weights_single_set(self, dft_dict_all, reg_dict, weight_stress=0.1):

        first_indices = reg_dict["first_indices"]
        x = reg_dict["x"]
        n_data, n_features = x.shape
        y = np.zeros(n_data)
        w = np.ones(n_data)

        if self.__multiple_datasets is False:
            indices = first_indices[0]
            x, y, w = apply_weight_percentage(
                x,
                y,
                w,
                dft_dict_all,
                self.__common_params_dict,
                indices,
                weight_stress=weight_stress,
                min_e=self.__min_energy,
            )
        else:
            for dft_dict, indices in zip(dft_dict_all.values(), first_indices):
                x, y, w = apply_weight_percentage(
                    x,
                    y,
                    w,
                    dft_dict,
                    self.__common_params_dict,
                    indices,
                    weight_stress=weight_stress,
                    min_e=self.__min_energy,
                )

        reg_dict["x"] = x
        reg_dict["y"] = y
        reg_dict["weight"] = w
        reg_dict["scales"] = self.__scales

        return reg_dict

    @abstractmethod
    def run(self):
        pass

    @property
    def params_dict(self):
        return self.__params_dict

    @property
    def common_params_dict(self):
        return self.__common_params_dict

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

    @property
    def min_energy(self):
        return self.__min_energy

    @property
    def train_regression_dict(self):
        return self.__train_reg_dict

    @property
    def test_regression_dict(self):
        return self.__test_reg_dict

    @train_regression_dict.setter
    def train_regression_dict(self, dict1):
        self.__train_reg_dict = dict1

    @test_regression_dict.setter
    def test_regression_dict(self, dict1):
        self.__test_reg_dict = dict1

    @property
    def features_class(self):
        return self.__feature_class

    @property
    def verbose(self):
        return self.__verbose
