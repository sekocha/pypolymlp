#!/usr/bin/env python
import os
import sys

import numpy as np

from pypolymlp.cxx.lib import libmlpcpp
from pypolymlp.mlp_gen.multi_datasets.features import multiple_dft_dicts_to_mlpcpp_obj


class Features:

    def __init__(
        self,
        multiple_params_dicts,
        multiple_dft_dicts,
        print_memory=True,
        element_swap=False,
    ):

        res = multiple_dft_dicts_to_mlpcpp_obj(multiple_dft_dicts)
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_sum_array,
            n_st_dataset,
            force_dataset,
        ) = res

        for params_dict in multiple_params_dicts:
            params_dict["element_swap"] = element_swap
            params_dict["print_memory"] = print_memory

        obj = libmlpcpp.PotentialAdditiveModel(
            multiple_params_dicts,
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            n_atoms_sum_array,
        )

        self.x = obj.get_x()
        self.fbegin, self.sbegin = obj.get_fbegin(), obj.get_sbegin()
        self.cumulative_n_features = obj.get_cumulative_n_features()
        self.ne, self.nf, self.ns = obj.get_n_data()

        self.ebegin, ei = [], 0
        for n in n_st_dataset:
            self.ebegin.append(ei)
            ei += n
        self.ebegin = np.array(self.ebegin)

        self.reg_dict = dict()
        self.reg_dict["x"] = self.x
        self.reg_dict["first_indices"] = list(
            zip(self.ebegin, self.fbegin, self.sbegin)
        )
        self.reg_dict["n_data"] = (self.ne, self.nf, self.ns)
        self.reg_dict["cumulative_n_features"] = self.cumulative_n_features

    def get_regression_dict(self):
        return self.reg_dict

    def get_x(self):
        return self.x

    def get_first_indices(self):
        return self.reg_dict["first_indices"]

    def get_cumulative_n_features(self):
        return self.reg_dict["cumulative_n_features"]

    def get_n_data(self):
        return self.reg_dict["n_data"]
