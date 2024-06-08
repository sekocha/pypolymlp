#!/usr/bin/env python
import numpy as np

from pypolymlp.mlp_gen.precondition import apply_atomic_energy, apply_weight_percentage


class Precondition:

    def __init__(
        self,
        reg_dict,
        multiple_dft_dicts,
        params_dict,
        scales=None,
        weight_stress=0.1,
    ):

        self.x = reg_dict["x"]
        self.first_indices = reg_dict["first_indices"]
        self.ne, self.nf, self.ns = reg_dict["n_data"]
        self.n_data, self.n_features = self.x.shape

        self.reg_dict = reg_dict
        self.multiple_dft_dicts = multiple_dft_dicts
        self.params_dict = params_dict

        self.y = np.zeros(self.n_data)
        self.w = np.ones(self.n_data)
        self.scales = None

        self.__apply_atomic_energy()
        min_e_per_atom = self.__find_min_energy()
        self.__apply_scales(scales=scales)
        self.__apply_weight(weight_stress=weight_stress, min_e=min_e_per_atom)

        self.reg_dict["x"] = self.x
        self.reg_dict["y"] = self.y
        self.reg_dict["weight"] = self.w
        self.reg_dict["scales"] = self.scales

    def __apply_atomic_energy(self):

        for _, dft_dict in self.multiple_dft_dicts.items():
            dft_dict = apply_atomic_energy(dft_dict, self.params_dict)

    def __find_min_energy(self):

        min_e = 1e10
        for _, dft_dict in self.multiple_dft_dicts.items():
            e_per_atom = dft_dict["energy"] / dft_dict["total_n_atoms"]
            min_e_trial = np.min(e_per_atom)
            if min_e_trial < min_e:
                min_e = min_e_trial
        return min_e

    def __apply_scales(self, scales=None):

        if scales is not None:
            self.scales = scales
        else:
            self.scales = np.std(self.x[: self.ne], axis=0)

        self.x /= self.scales
        """ correctly-working numba version
        numba_support.mat_prod_vec(self.x, np.reciprocal(self.scales), axis=1)
        """

    def __apply_weight(self, weight_stress=0.1, min_e=None):

        for (_, dft_dict), indices in zip(
            self.multiple_dft_dicts.items(), self.first_indices
        ):
            res = apply_weight_percentage(
                self.x,
                self.y,
                self.w,
                dft_dict,
                self.params_dict,
                indices,
                weight_stress=weight_stress,
                min_e=min_e,
            )
            self.x, self.y, self.w = res

    def print_data_shape(self, header="training data size"):

        print("  " + header + ":", self.x.shape)
        print("   - n (energy) =", self.ne)
        print("   - n (force)  =", self.nf)
        print("   - n (stress) =", self.ns)

    def get_scales(self):
        return self.scales

    def get_updated_regression_dict(self):
        return self.reg_dict
