#!/usr/bin/env python
import glob
import itertools

import numpy as np
from setuptools._distutils.util import strtobool

from pypolymlp.core.parser_infile import InputParser
from pypolymlp.cxx.lib import libmlpcpp


class ParamsParser:

    def __init__(
        self,
        filename,
        multiple_datasets=False,
        parse_vasprun_locations=True,
    ):

        self.parser = InputParser(filename)

        params = dict()
        params["n_type"] = self.parser.get_params("n_type", default=1, dtype=int)
        params["include_force"] = self.parser.get_params(
            "include_force", default=True, dtype=bool
        )
        if params["include_force"]:
            params["include_stress"] = self.parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            params["include_stress"] = False

        self.n_type = params["n_type"]
        self.include_force = params["include_force"]

        params["model"] = self.__get_potential_model_params(params["n_type"])
        params["atomic_energy"] = self.__get_atomic_energy(params["n_type"])
        params["reg"] = self.__get_regression_params()

        """ DFT data locations"""
        params["dataset_type"] = self.parser.get_params("dataset_type", default="vasp")
        if parse_vasprun_locations:
            if params["dataset_type"] == "vasp":
                if multiple_datasets:
                    params["dft"] = self.__get_multiple_vasprun_sets()
                else:
                    params["dft"] = self.__get_single_vasprun_set()
            elif params["dataset_type"] == "phono3py":
                params["dft"] = self.__get_phono3py_set()

        params["elements"] = self.parser.get_params(
            "elements",
            size=params["n_type"],
            default=None,
            required=True,
            dtype=str,
            return_array=True,
        )
        rearrange = self.parser.get_params(
            "rearrange_by_elements", default=True, dtype=bool
        )
        params["element_order"] = params["elements"] if rearrange else None

        self.params_dict = params

    def __get_potential_model_params(self, n_type):

        model = dict()
        model["cutoff"] = self.parser.get_params("cutoff", default=6.0, dtype=float)
        model["model_type"] = self.parser.get_params("model_type", default=1, dtype=int)
        model["max_p"] = self.parser.get_params("max_p", default=1, dtype=int)
        model["feature_type"] = self.parser.get_params("feature_type", default="gtinv")

        gtinv_dict = dict()
        if model["feature_type"] == "gtinv":
            gtinv_dict["order"] = self.parser.get_params(
                "gtinv_order", default=3, dtype=int
            )
            size = gtinv_dict["order"] - 1
            d_maxl = [2 for i in range(size)]
            gtinv_dict["max_l"] = self.parser.get_params(
                "gtinv_maxl",
                size=size,
                default=d_maxl,
                dtype=int,
                return_array=True,
            )
            if len(gtinv_dict["max_l"]) < size:
                size_gap = size - len(gtinv_dict["max_l"])
                for i in range(size_gap):
                    gtinv_dict["max_l"].append(2)

            gtinv_sym = [False for i in range(size)]

            gtinv_dict["version"] = self.parser.get_params(
                "gtinv_version", default=1, dtype=int
            )
            rgi = libmlpcpp.Readgtinv(
                gtinv_dict["order"],
                gtinv_dict["max_l"],
                gtinv_sym,
                n_type,
                gtinv_dict["version"],
            )
            gtinv_dict["lm_seq"] = rgi.get_lm_seq()
            gtinv_dict["l_comb"] = rgi.get_l_comb()
            gtinv_dict["lm_coeffs"] = rgi.get_lm_coeffs()
            model["max_l"] = max(gtinv_dict["max_l"])
        else:
            gtinv_dict["order"] = 0
            gtinv_dict["max_l"] = []
            gtinv_dict["lm_seq"] = []
            gtinv_dict["l_comb"] = []
            gtinv_dict["lm_coeffs"] = []
            gtinv_dict["version"] = 1
            model["max_l"] = 0
        model["gtinv"] = gtinv_dict

        model["pair_type"] = "gaussian"
        d_params1 = [1.0, 1.0, 1]
        params1 = self.parser.get_sequence("gaussian_params1", default=d_params1)
        d_params2 = [0.0, model["cutoff"] - 1.0, 7]
        params2 = self.parser.get_sequence("gaussian_params2", default=d_params2)
        model["pair_params"] = list(itertools.product(params1, params2))
        model["pair_params"].append([0.0, 0.0])

        return model

    def __get_atomic_energy(self, n_type):

        d_atom_e = [0.0 for i in range(n_type)]
        atom_e = self.parser.get_params(
            "atomic_energy",
            size=n_type,
            default=d_atom_e,
            dtype=float,
            return_array=True,
        )
        return atom_e

    def __get_regression_params(self):

        reg = dict()
        reg["method"] = "ridge"
        d_alpha = [-3, 1, 5]
        reg["alpha"] = self.parser.get_sequence("reg_alpha_params", default=d_alpha)
        return reg

    def __get_single_vasprun_set(self):

        train = self.parser.get_params("train_data", default=None)
        test = self.parser.get_params("test_data", default=None)

        data = dict()
        data["train"] = sorted(glob.glob(train))
        data["test"] = sorted(glob.glob(test))
        return data

    def __get_multiple_vasprun_sets(self):

        train = self.parser.get_train()
        test = self.parser.get_test()

        for params in train:
            shortage = []
            if len(params) < 2:
                shortage.append("True")
            if len(params) < 3:
                shortage.append(1.0)
            params.extend(shortage)

        for params in test:
            shortage = []
            if len(params) < 2:
                shortage.append("True")
            if len(params) < 3:
                shortage.append(1.0)
            params.extend(shortage)

        if self.include_force is False:
            for params in train:
                params[1] = "False"
            for params in test:
                params[1] = "False"

        data = dict()
        data["train"], data["test"] = dict(), dict()
        for params in train:
            set_id = params[0]
            data["train"][set_id] = dict()
            data["train"][set_id]["vaspruns"] = sorted(glob.glob(set_id))
            data["train"][set_id]["include_force"] = strtobool(params[1])
            data["train"][set_id]["weight"] = float(params[2])
        for params in test:
            set_id = params[0]
            data["test"][set_id] = dict()
            data["test"][set_id]["vaspruns"] = sorted(glob.glob(set_id))
            data["test"][set_id]["include_force"] = strtobool(params[1])
            data["test"][set_id]["weight"] = float(params[2])
        return data

    def __get_phono3py_set(self):
        """
        Format
        ------
        1.
        phono3py_train_data phono3py_params.yaml.xz energies.dat
        phono3py_test_data phono3py_params.yaml.xz energies.dat
        2.
        phono3py_train_data phono3py_params.yaml.xz energies.dat 0 200
        phono3py_test_data phono3py_params.yaml.xz energies.dat 950 1000
        3.
        phono3py_train_data phono3py_params.yaml.xz
        phono3py_test_data phono3py_params.yaml.xz
        4.
        phono3py_train_data phono3py_params.yaml.xz 0 200
        phono3py_test_data phono3py_params.yaml.xz 950 1000
        """
        train = self.parser.get_params("phono3py_train_data", size=4, default=None)
        test = self.parser.get_params("phono3py_test_data", size=4, default=None)
        phono3py_sample = self.parser.get_params("phono3py_sample", default="sequence")

        data = dict()
        data["train"] = dict()
        data["test"] = dict()
        data["train"]["phono3py_yaml"] = train[0]
        data["test"]["phono3py_yaml"] = test[0]

        if len(train) == 2 or len(train) == 4:
            data["train"]["energy"] = train[1]
            data["test"]["energy"] = test[1]

        if len(train) > 2:
            if phono3py_sample == "sequence":
                data["train"]["indices"] = np.arange(int(train[-2]), int(train[-1]))
            elif phono3py_sample == "random":
                data["train"]["indices"] = np.random.choice(
                    int(train[-2]), size=int(train[-1])
                )
        else:
            data["train"]["indices"] = None

        if len(test) > 2:
            data["test"]["indices"] = np.arange(int(test[-2]), int(test[-1]))
        else:
            data["test"]["indices"] = None

        return data

    def get_params(self):
        return self.params_dict
