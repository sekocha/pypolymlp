#!/usr/bin/env python
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase


class PolymlpDevDataXYEnsembleBase(PolymlpDevDataXYBase):

    def __init__(self, params: PolymlpDevData):

        super().__init__(params)

        if not self.is_multiple_datasets:
            raise ValueError(
                "Ensemble version is available "
                "for PolymlpDevParams with multiple datasets."
            )

        self.__random_indices = None
        self.__n_models = None
        self.__n_features = None
        self.__train_regression_dict_list = None
        self.__test_regression_dict_list = None
        self.__cumulative_n_features = None
        self.__scales_list = None

    @property
    def random_indices(self):
        return self.__random_indices

    @random_indices.setter
    def random_indices(self, i):
        self.__random_indices = i

    @property
    def n_models(self):
        return self.__n_models

    @n_models.setter
    def n_models(self, i):
        self.__n_models = i

    @property
    def n_features(self):
        return self.__n_features

    @n_features.setter
    def n_features(self, i):
        self.__n_features = i

    @property
    def train_regression_dict_list(self):
        """
        Keys in reg_dict
        ----------------
        - x.T @ X
        - x.T @ y
        - y_sq_norm,
        - scales
        - total_n_data,
        """
        return self.__train_regression_dict_list

    @train_regression_dict_list.setter
    def train_regression_dict_list(self, i):
        self.__train_regression_dict_list = i

    @property
    def test_regression_dict_list(self):
        return self.__test_regression_dict_list

    @test_regression_dict_list.setter
    def test_regression_dict_list(self, i):
        self.__test_regression_dict_list = i

    @property
    def cumulative_n_features(self):
        return self.__cumulative_n_features

    @cumulative_n_features.setter
    def cumulative_n_features(self, i):
        self.__cumulative_n_features = i

    @property
    def scales_list(self):
        return self.__scales_list

    @scales_list.setter
    def scales_list(self, i):
        self.__scales_list = i
