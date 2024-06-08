#!/usr/bin/env python
import os
import sys

import numpy as np

from pypolymlp.cxx.lib import libmlpcpp


def structures_to_mlpcpp_obj(structures):
    axis_array = [st["axis"] for st in structures]
    positions_c_array = [np.dot(st["axis"], st["positions"]) for st in structures]
    types_array = [st["types"] for st in structures]
    n_atoms_sum_array = [sum(st["n_atoms"]) for st in structures]
    return (axis_array, positions_c_array, types_array, n_atoms_sum_array)


class Features:

    def __init__(self, params_dict, structures, print_memory=True, element_swap=False):

        n_st_dataset = [len(structures)]
        force_dataset = [params_dict["include_force"]]
        res = structures_to_mlpcpp_obj(structures)
        (
            axis_array,
            positions_c_array,
            types_array,
            self.n_atoms_sum_array,
        ) = res

        params_dict["element_swap"] = element_swap
        params_dict["print_memory"] = print_memory
        obj = libmlpcpp.PotentialModel(
            params_dict,
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            self.n_atoms_sum_array,
        )
        self.x = obj.get_x()
        self.fbegin, self.sbegin = obj.get_fbegin(), obj.get_sbegin()
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

    def get_regression_dict(self):
        return self.reg_dict

    def get_x(self):
        return self.x

    def get_first_indices(self):
        return self.reg_dict["first_indices"]

    def get_n_data(self):
        return self.reg_dict["n_data"]

    def get_n_atoms_sums(self):
        return self.n_atoms_sum_array
