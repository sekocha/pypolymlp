#!/usr/bin/env python 
import numpy as np

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.core.interface_vasp import parse_vaspruns

from pypolymlp.mlp_dev.mlpdev_core import PolymlpDevParams

class PolymlpDev:

    def __init__(self, params: PolymlpDevParams):

        self.__params_dict = params.params_dict
        self.__hybrid_params_dicts = params.hybrid_params_dicts

        self.__train_dict = params.train_dict
        self.__test_dict = params.test_dict
        self.__min_energy = params.min_energy

        self.__train_reg_dict = None
        self.__test_reg_dict = None

    def compute_features(self):

        f_obj = Features(self.__params_dict, self.__train_dict)
        self.__train_reg_dict = f_obj.regression_dict

        f_obj = Features(self.__params_dict, self.__test_reg_dict)
        self.__test_reg_dict = f_obj.regression_dict

    def apply_weights(self):
        pass

    def apply_scales(self):
        pass

    def run_sequential(self):
        pass

    @property
    def params_dict(self):
        return self.__params_dict

    @property
    def hybrid_params_dicts(self):
        return self.__hybrid_params_dicts

    @property
    def train_dict(self):
        return self.__train_dict

    @property
    def test_dict(self):
        return self.__test_dict


