#!/usr/bin/env python
import copy

import numpy as np

from pypolymlp.core.interface_vasp import parse_vaspruns
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_dev.core.features_attr import write_polymlp_params_yaml


def get_variable_with_max_length(multiple_params_dicts, key):

    array = []
    for single in multiple_params_dicts:
        if len(single[key]) > len(array):
            array = single[key]
    return array


def set_common_params_dict(multiple_params_dicts):

    keys = set()
    for single in multiple_params_dicts:
        for k in single.keys():
            keys.add(k)

    common_params_dict = copy.copy(multiple_params_dicts[0])

    n_type = max([single["n_type"] for single in multiple_params_dicts])

    elements = get_variable_with_max_length(multiple_params_dicts, "elements")
    bool_element_order = [
        single["element_order"] for single in multiple_params_dicts
    ] is not None
    element_order = elements if bool_element_order else None

    atom_e = get_variable_with_max_length(multiple_params_dicts, "atomic_energy")

    common_params_dict["n_type"] = n_type
    common_params_dict["elements"] = elements
    common_params_dict["element_order"] = element_order
    common_params_dict["atomic_energy"] = atom_e

    return common_params_dict


class PolymlpDevData:
    """
    Variables in params_dict
    ------------------------
      - n_type
      - include_force
      - include_stress
      - model
        - cutoff
        - model_type
        - max_p
        - max_l
        - feature_type
        - pair_type
        - pair_params
        - gtinv
          - order
          - max_l
          - lm_seq
          - l_comb
          - lm_coeffs
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train (vasprun locations)
        - test (vasprun locations)

    Variables in dft_dict (train_dft_dict, test_dft_dict)
    -----------------------------------------------------
        - energy
        - force
        - stress
        - structures
          - structure (1)
            - axis
            - positions
            - n_atoms
            - types
            - elements
          - ...
        - elements
        - volumes
        - total_n_atoms
    """

    def __init__(self):

        self.__params_dict = None
        self.__hybrid_params_dicts = None

        self.__train_dict = None
        self.__test_dict = None

        self.__multiple_datasets = None
        self.__hybrid = False

    def parse_infiles(self, infiles, verbose=True):

        if isinstance(infiles, list) is False:
            p = ParamsParser(infiles, multiple_datasets=True)
            self.__params_dict = p.get_params()
            priority_infile = infiles
        else:
            priority_infile = infiles[0]
            if len(infiles) == 1:
                p = ParamsParser(priority_infile, multiple_datasets=True)
                self.__params_dict = p.get_params()
            else:
                self.__hybrid_params_dicts = [
                    ParamsParser(infile, multiple_datasets=True).get_params()
                    for infile in infiles
                ]
                self.__params_dict = set_common_params_dict(self.__hybrid_params_dicts)
                self.__hybrid = True

        if verbose:
            self.print_params(infile=priority_infile)

        return self

    def parse_datasets(self):
        """todo: Must be revised"""
        if "phono3py_yaml" in self.__params_dict["dft"]["train"]:
            self.parse_single_dataset()
            self.__params_dict["dft"]["train"]["train1"] = self.__params_dict["dft"][
                "train"
            ]
            self.__params_dict["dft"]["test"]["test1"] = self.__params_dict["dft"][
                "test"
            ]
            self.__train_dict = {"train1": self.__train_dict}
            self.__test_dict = {"test1": self.__test_dict}
            self.__multiple_datasets = True
        else:
            self.parse_multiple_datasets()
            self.__multiple_datasets = True

    def parse_single_dataset(self):

        if self.__params_dict is None:
            raise ValueError("parse_dataset: params_dict is needed.")

        dataset_type = self.__params_dict["dataset_type"]
        if dataset_type == "vasp":
            self.__train_dict = parse_vaspruns(
                self.__params_dict["dft"]["train"],
                element_order=self.__params_dict["element_order"],
            )
            self.__test_dict = parse_vaspruns(
                self.__params_dict["dft"]["test"],
                element_order=self.__params_dict["element_order"],
            )
        elif dataset_type == "phono3py":
            from pypolymlp.core.interface_phono3py_ver3 import parse_phono3py_yaml

            self.__train_dict = parse_phono3py_yaml(
                self.__params_dict["dft"]["train"]["phono3py_yaml"],
                self.__params_dict["dft"]["train"]["energy"],
                element_order=self.__params_dict["element_order"],
                select_ids=self.__params_dict["dft"]["train"]["indices"],
                use_phonon_dataset=False,
            )
            self.__test_dict = parse_phono3py_yaml(
                self.__params_dict["dft"]["test"]["phono3py_yaml"],
                self.__params_dict["dft"]["test"]["energy"],
                element_order=self.__params_dict["element_order"],
                select_ids=self.__params_dict["dft"]["test"]["indices"],
                use_phonon_dataset=False,
            )

        self.__train_dict = self.__apply_atomic_energy(self.__train_dict)
        self.__test_dict = self.__apply_atomic_energy(self.__test_dict)
        return self

    def parse_multiple_datasets(self):

        if self.__params_dict is None:
            raise ValueError("parse_dataset: params_dict is needed.")

        dataset_type = self.__params_dict["dataset_type"]
        if dataset_type == "vasp":
            element_order = self.__params_dict["element_order"]
            self.__train_dict, self.__test_dict = dict(), dict()
            for set_id, dict1 in self.__params_dict["dft"]["train"].items():
                self.__train_dict[set_id] = parse_vaspruns(
                    dict1["vaspruns"], element_order=element_order
                )
                self.__train_dict[set_id].update(dict1)

            for set_id, dict1 in self.__params_dict["dft"]["test"].items():
                self.__test_dict[set_id] = parse_vaspruns(
                    dict1["vaspruns"], element_order=element_order
                )
                self.__test_dict[set_id].update(dict1)
        else:
            raise KeyError("Only dataset_type = vasp is available.")

        for _, dft_dict in self.__train_dict.items():
            dft_dict = self.__apply_atomic_energy(dft_dict)
        for _, dft_dict in self.__test_dict.items():
            dft_dict = self.__apply_atomic_energy(dft_dict)

        self.__multiple_datasets = True

        return self

    def __apply_atomic_energy(self, dft_dict):

        energy = dft_dict["energy"]
        structures = dft_dict["structures"]
        atom_e = self.__params_dict["atomic_energy"]
        coh_energy = [
            e - np.dot(st["n_atoms"], atom_e) for e, st in zip(energy, structures)
        ]
        dft_dict["energy"] = np.array(coh_energy)
        return dft_dict

    def print_params(self, infile=None):

        if infile is not None:
            print("priority_input:", infile)

        params_dict = self.common_params_dict
        print("parameters:")
        print("  n_types:       ", params_dict["n_type"])
        print("  elements:      ", params_dict["elements"])
        print("  element_order: ", params_dict["element_order"])
        print("  atomic_energy: ", params_dict["atomic_energy"])
        print("  include_force: ", bool(params_dict["include_force"]))
        print("  include_stress:", bool(params_dict["include_stress"]))

        """ todo: Must be revised"""
        if self.is_multiple_datasets is not None:
            if self.is_multiple_datasets:
                print("  train_data:")
                for v in params_dict["dft"]["train"]:
                    print("  -", v)
                print("  test_data:")
                for v in params_dict["dft"]["test"]:
                    print("  -", v)
            else:
                print("  train_data:")
                print("  -", params_dict["dft"]["train"]["phono3py_yaml"])
                print("  test_data:")
                print("  -", params_dict["dft"]["test"]["phono3py_yaml"])

    def write_polymlp_params_yaml(self, filename="polymlp_params.yaml"):

        if not self.is_hybrid:
            write_polymlp_params_yaml(self.params_dict, filename=filename)
        else:
            for i, params in enumerate(self.params_dict):
                filename = "polymlp_params" + str(i + 1) + ".yaml"
                write_polymlp_params_yaml(params, filename=filename)

    @property
    def params_dict(self):
        if self.__hybrid:
            return self.__hybrid_params_dicts
        return self.__params_dict

    @property
    def common_params_dict(self):
        return self.__params_dict

    @property
    def hybrid_params_dicts(self):
        return self.__hybrid_params_dicts

    @params_dict.setter
    def params_dict(self, params):
        if isinstance(params, list):
            if len(params) > 1:
                self.hybrid_params_dicts = params
            else:
                self.__params_dict = params[0]
                self.__hybrid = False
        else:
            self.__params_dict = params
            self.__hybrid = False

    @hybrid_params_dicts.setter
    def hybrid_params_dicts(self, params):
        self.__hybrid_params_dicts = params
        self.__params_dict = set_common_params_dict(params)
        self.__hybrid = True

    @property
    def train_dict(self):
        return self.__train_dict

    @property
    def test_dict(self):
        return self.__test_dict

    @train_dict.setter
    def train_dict(self, dict1):
        self.__train_dict = dict1
        if "structures" in dict1:
            self.__multiple_datasets = False
        else:
            self.__multiple_datasets = True

    @test_dict.setter
    def test_dict(self, dict1):
        self.__test_dict = dict1
        if "structures" in dict1:
            self.__multiple_datasets = False
        else:
            self.__multiple_datasets = True

    @property
    def is_multiple_datasets(self):
        return self.__multiple_datasets

    @property
    def is_hybrid(self):
        return self.__hybrid

    @property
    def min_energy(self):

        if self.__multiple_datasets:
            min_e = 1e10
            for dft_dict in self.__train_dict.values():
                e_per_atom = dft_dict["energy"] / dft_dict["total_n_atoms"]
                min_e_trial = np.min(e_per_atom)
                if min_e_trial < min_e:
                    min_e = min_e_trial
        else:
            dft_dict = self.__train_dict
            e_per_atom = dft_dict["energy"] / dft_dict["total_n_atoms"]
            min_e = np.min(e_per_atom)

        return min_e
