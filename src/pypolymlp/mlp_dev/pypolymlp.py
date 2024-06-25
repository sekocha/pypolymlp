#!/usr/bin/env python
import itertools

import numpy as np

from pypolymlp.core.displacements import convert_disps_to_positions, set_dft_dict
from pypolymlp.cxx.lib import libmlpcpp
from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression


class Pypolymlp:

    def __init__(self):
        """
        Keys in params_dict
        --------------------
        - n_type
        - include_force
        - include_stress
        - atomic_energy
        - dataset_type
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
        - reg
          - method
          - alpha
        - dft
          - train (dataset locations)
          - test (dataset locations)
        """
        self.__params_dict = dict()
        self.__params_dict["model"] = dict()
        self.__params_dict["model"]["gtinv"] = dict()
        self.__params_dict["reg"] = dict()
        self.__params_dict["dft"] = dict()
        self.__params_dict["dft"]["train"] = dict()
        self.__params_dict["dft"]["test"] = dict()

        self.train_dft_dict = None
        self.test_dft_dict = None

        self.__mlp_dict = None
        self.__multiple_datasets = False

        """Hybrid models are not available."""
        # self.__hybrid = None

    def __set_param(self, tag_params, params, assign_variable):

        if params is not None and tag_params in params:
            assign_variable = params[tag_params]
        return assign_variable

    def set_params(
        self,
        elements=None,
        params=None,
        include_force=True,
        include_stress=False,
        cutoff=6.0,
        model_type=4,
        max_p=2,
        feature_type="gtinv",
        gaussian_params1=(1.0, 1.0, 1),
        gaussian_params2=(0.0, 5.0, 7),
        reg_alpha_params=(-3.0, 1.0, 5),
        gtinv_order=3,
        gtinv_maxl=(4, 4, 2, 1, 1),
        gtinv_version=1,
        atomic_energy=None,
        rearrange_by_elements=True,
    ):
        """
        Assign input parameters.

        Parameters
        ----------
        elements: Element species, (e.g., ['Mg','O'])
        include_force: Considering force entries
        include_stress: Considering stress entries
        cutoff: Cutoff radius
        model_type: Polynomial function type
            model_type = 1: Linear polynomial of polynomial invariants
            model_type = 2: Polynomial of polynomial invariants
            model_type = 3: Polynomial of pair invariants
                            + linear polynomial of polynomial invariants
            model_type = 4: Polynomial of pair and second-order invariants
                            + linear polynomial of polynomial invariants
        max_p: Order of polynomial function
        feature_type: 'gtinv' or 'pair'
        gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
            Parameters are given as np.linspace(p[0], p[1], p[2]),
            where p[0], p[1], and p[2] are given by gaussian_params1
            and gaussian_params2.
        reg_alpha_params: Parameters for penalty term in
            linear ridge regression. Parameters are given as
            np.linspace(p[0], p[1], p[2]).
        gtinv_order: Maximum order of polynomial invariants.
        gtinv_maxl: Maximum angular numbers of polynomial invariants.
            [maxl for order=2, maxl for order=3, ...]
        atomic_energy: Atomic energies.
        rearrange_by_elements: Set True if not developing special MLPs.

        All parameters are stored in self.__params_dict.
        """
        if params is None:
            params = dict()

        self.__params_dict["elements"] = self.__set_param("elements", params, elements)
        if self.__params_dict["elements"] is None:
            raise ValueError("elements must be provided.")

        n_type = len(self.__params_dict["elements"])
        self.__params_dict["n_type"] = n_type

        self.__params_dict["include_force"] = self.__set_param(
            "include_force", params, include_force
        )
        self.__params_dict["include_stress"] = self.__set_param(
            "include_stress", params, include_stress
        )

        model = self.__params_dict["model"]
        model["cutoff"] = self.__set_param("cutoff", params, cutoff)
        model["model_type"] = self.__set_param("model_type", params, model_type)
        if model["model_type"] > 4:
            raise ValueError("model_type != 1, 2, 3, or 4")

        model["max_p"] = self.__set_param("max_p", params, max_p)
        if model["max_p"] > 3:
            raise ValueError("model_type != 1, 2, or 3")

        model["feature_type"] = self.__set_param("feature_type", params, feature_type)
        if model["feature_type"] != "gtinv" and model["feature_type"] != "pair":
            raise ValueError("feature_type != gtinv or pair")

        model["pair_type"] = "gaussian"

        gaussian_params1 = self.__set_param(
            "gaussian_params1", params, gaussian_params1
        )
        gaussian_params2 = self.__set_param(
            "gaussian_params2", params, gaussian_params2
        )
        if len(gaussian_params1) != 3:
            raise ValueError("len(gaussian_params1) != 3")
        if len(gaussian_params2) != 3:
            raise ValueError("len(gaussian_params2) != 3")
        params1 = self.__sequence(gaussian_params1)
        params2 = self.__sequence(gaussian_params2)
        model["pair_params"] = list(itertools.product(params1, params2))
        model["pair_params"].append([0.0, 0.0])

        gtinv_dict = self.__params_dict["model"]["gtinv"]
        if model["feature_type"] == "gtinv":
            gtinv_dict["order"] = self.__set_param("gtinv_order", params, gtinv_order)
            gtinv_dict["max_l"] = self.__set_param("gtinv_maxl", params, gtinv_maxl)
            gtinv_dict["max_l"] = list(gtinv_dict["max_l"])

            size = gtinv_dict["order"] - 1
            if len(gtinv_dict["max_l"]) < size:
                raise ValueError("size (gtinv_maxl) !=", size)

            gtinv_sym = [False for i in range(size)]
            gtinv_dict["version"] = self.__set_param(
                "gtinv_version", params, gtinv_version
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
            model["max_l"] = 0

        reg_alpha_params = self.__set_param(
            "reg_alpha_params", params, reg_alpha_params
        )
        if len(reg_alpha_params) != 3:
            raise ValueError("len(reg_alpha_params) != 3")
        self.__params_dict["reg"]["method"] = "ridge"
        self.__params_dict["reg"]["alpha"] = self.__sequence(reg_alpha_params)

        atomic_energy = self.__set_param("atomic_energy", params, atomic_energy)
        if atomic_energy is None:
            self.__params_dict["atomic_energy"] = [0.0 for i in range(n_type)]
        else:
            if len(atomic_energy) != n_type:
                raise ValueError("len(atomic_energy) != n_type")
            self.__params_dict["atomic_energy"] = atomic_energy

        if rearrange_by_elements:
            self.__params_dict["element_order"] = self.__params_dict["elements"]
        else:
            self.__params_dict["element_order"] = None
        return self

    def set_datasets_vasp(self, train_vaspruns, test_vaspruns):
        """
        Parameters
        ----------
        train_vaspruns: vasprun files for training dataset (list)
        test_vaspruns: vasprun files for test dataset (list)
        """
        self.__params_dict["dataset_type"] = "vasp"
        self.__params_dict["dft"]["train"] = sorted(train_vaspruns)
        self.__params_dict["dft"]["test"] = sorted(test_vaspruns)
        return self

    def set_multiple_datasets_vasp(self, train_vaspruns, test_vaspruns):
        """
        Parameters
        ----------
        train_vaspruns: list of list containing vasprun files (training)
        test_vaspruns: list of list containing vasprun files (test)
        """
        self.__params_dict["dataset_type"] = "vasp"
        self.__params_dict["dft"]["train"] = dict()
        self.__params_dict["dft"]["test"] = dict()
        for i, vaspruns in enumerate(train_vaspruns):
            self.__params_dict["dft"]["train"]["dataset" + str(i + 1)] = {
                "vaspruns": sorted(vaspruns),
                "include_force": True,
                "weight": 1.0,
            }

        for i, vaspruns in enumerate(test_vaspruns):
            self.__params_dict["dft"]["test"]["dataset" + str(i + 1)] = {
                "vaspruns": sorted(vaspruns),
                "include_force": True,
                "weight": 1.0,
            }
        self.__multiple_datasets = True
        return self

    def set_datasets_phono3py(
        self,
        train_yaml,
        test_yaml,
        train_energy_dat=None,
        test_energy_dat=None,
        train_ids=None,
        test_ids=None,
    ):

        self.__params_dict["dataset_type"] = "phono3py"
        data = self.__params_dict["dft"]
        data["train"], data["test"] = dict(), dict()
        data["train"]["phono3py_yaml"] = train_yaml
        data["train"]["energy"] = train_energy_dat
        data["test"]["phono3py_yaml"] = test_yaml
        data["test"]["energy"] = test_energy_dat

        data["train"]["indices"] = train_ids
        data["test"]["indices"] = test_ids
        return self

    def set_datasets_displacements(
        self,
        train_disps,
        train_forces,
        train_energies,
        test_disps,
        test_forces,
        test_energies,
        structure_without_disp,
    ):
        """Set datasets from displacements-(energies, forces).

        Parameters
        ----------
        train_disps: displacements (training data),
                     shape=(n_train, 3, n_atoms)
        train_forces: forces (training data),
                      shape=(n_train, 3, n_atoms)
        train_energies: energies (training data),
                      shape=(n_train)
        test_disps: displacements (test data),
                    shape=(n_test, 3, n_atom)
        test_forces: forces (test data),
                    shape=(n_test, 3, n_atom)
        test_energies: energies (test data),
                    shape=(n_test)

        structure_without_disp: structure without displacements, dict
        - 'axis': (3,3), [a, b, c]
        - 'positions': (3, n_atom) [x1, x2, ...]
        - 'n_atoms': [4, 4]
        - 'elements': Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
        - 'types': Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
        - 'volume': 64.0 (ang.^3)
        """
        try:
            element_order = self.__params_dict["elements"]
        except KeyError:
            raise KeyError(
                "Set parameters using set_params() "
                "before using set_datasets_displacements."
            )

        self.train_dft_dict = self.__set_dft_dict(
            train_disps,
            train_forces,
            train_energies,
            structure_without_disp,
            element_order=element_order,
        )
        self.test_dft_dict = self.__set_dft_dict(
            test_disps,
            test_forces,
            test_energies,
            structure_without_disp,
            element_order=element_order,
        )
        return self

    def __set_dft_dict(
        self,
        disps,
        forces,
        energies,
        st_dict,
        element_order=None,
    ):

        positions_all = convert_disps_to_positions(
            disps, st_dict["axis"], st_dict["positions"]
        )
        dft_dict = set_dft_dict(
            forces,
            energies,
            positions_all,
            st_dict,
            element_order=element_order,
        )
        return dft_dict

    def __sequence(self, params):
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    def run(
        self,
        file_params=None,
        sequential=False,
        path_output="./",
        verbose=False,
        output_files=False,
    ):
        """Running linear ridge regression to estimate MLP coefficients."""
        polymlp_in = PolymlpDevData()
        if file_params is not None:
            polymlp_in.parse_infiles(file_params, verbose=True)
            self.__params_dict = polymlp_in.params_dict
        else:
            polymlp_in.params_dict = self.__params_dict

        if self.train_dft_dict is None:
            polymlp_in.parse_datasets()
        else:
            polymlp_in.train_dict = self.train_dft_dict
            polymlp_in.test_dict = self.test_dft_dict

        if output_files:
            polymlp_in.write_polymlp_params_yaml(
                filename=path_output + "/polymlp_params.yaml"
            )

        if sequential:
            polymlp = PolymlpDevDataXYSequential(polymlp_in, verbose=verbose).run()
        else:
            polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
            if verbose:
                polymlp.print_data_shape()

        reg = Regression(polymlp, verbose=verbose).fit(seq=sequential)
        self.__mlp_dict = reg.best_model
        if output_files:
            reg.save_mlp_lammps(filename=path_output + "/polymlp.lammps")

        acc = PolymlpDevAccuracy(reg)
        acc.compute_error(
            path_output=path_output,
            verbose=verbose,
            log_energy=output_files,
        )
        if output_files:
            acc.write_error_yaml(filename=path_output + "/polymlp_error.yaml")

        self.__mlp_dict["error"] = {
            "train": acc.error_train_dict,
            "test": acc.error_test_dict,
        }

        return self

    @property
    def parameters(self):
        return self.__params_dict

    @property
    def summary(self):
        return self.__mlp_dict

    @property
    def coeffs(self):
        return self.__mlp_dict["coeffs"] / self.__mlp_dict["scales"]
