"""Class of input file parser."""

import glob
import itertools
from typing import Literal

import numpy as np
from setuptools._distutils.util import strtobool

from pypolymlp.core.data_format import (
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParams,
)
from pypolymlp.core.parser_infile import InputParser


class ParamsParser:
    """Class of input file parser."""

    def __init__(
        self,
        filename: str,
        multiple_datasets: bool = False,
        parse_vasprun_locations: bool = True,
        prefix: str = None,
    ):
        """Init class.

        Parameters
        ----------
        filename: File of input parameters for single polymlp (e.g., polymlp.in).
        """

        self.parser = InputParser(filename)

        n_type = self.parser.get_params("n_type", default=1, dtype=int)
        elements = self.parser.get_params(
            "elements",
            size=n_type,
            default=None,
            required=True,
            dtype=str,
            return_array=True,
        )
        rearrange = self.parser.get_params(
            "rearrange_by_elements", default=True, dtype=bool
        )
        element_order = elements if rearrange else None
        self._elements = elements
        if element_order is not None:
            self._atomtypes = dict()
            for i, ele in enumerate(element_order):
                self._atomtypes[ele] = i

        self.include_force = include_force = self.parser.get_params(
            "include_force", default=True, dtype=bool
        )
        if self.include_force:
            include_stress = self.parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            include_stress = False

        atomic_energy = self._get_atomic_energy(n_type)
        reg_method, reg_alpha = self._get_regression_params()
        model = self._get_potential_model_params(n_type)

        if parse_vasprun_locations:
            dataset_type = self.parser.get_params("dataset_type", default="vasp")
            dft_train, dft_test = self._get_dataset(
                dataset_type, multiple_datasets, prefix=prefix
            )
        else:
            dataset_type = "vasp"
            dft_train, dft_test = None, None

        self._params = PolymlpParams(
            n_type=n_type,
            elements=elements,
            model=model,
            atomic_energy=atomic_energy,
            dft_train=dft_train,
            dft_test=dft_test,
            regression_method=reg_method,
            regression_alpha=reg_alpha,
            include_force=include_force,
            include_stress=include_stress,
            dataset_type=dataset_type,
            element_order=element_order,
        )

    def _get_potential_model_params(self, n_type: int):

        cutoff = self.parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self.parser.get_params("model_type", default=1, dtype=int)
        max_p = self.parser.get_params("max_p", default=1, dtype=int)
        feature_type = self.parser.get_params("feature_type", default="gtinv")

        if feature_type == "gtinv":
            order = self.parser.get_params("gtinv_order", default=3, dtype=int)
            size = order - 1
            d_maxl = [2 for i in range(size)]
            gtinv_maxl = self.parser.get_params(
                "gtinv_maxl",
                size=size,
                default=d_maxl,
                dtype=int,
                return_array=True,
            )
            if len(gtinv_maxl) < size:
                size_gap = size - len(gtinv_maxl)
                for i in range(size_gap):
                    gtinv_maxl.append(2)

            version = self.parser.get_params("gtinv_version", default=1, dtype=int)

            max_l = max(gtinv_maxl)
            gtinv_params = PolymlpGtinvParams(
                order=order,
                max_l=gtinv_maxl,
                n_type=n_type,
                version=version,
            )
        else:
            max_l = 0
            gtinv_params = PolymlpGtinvParams(order=0, max_l=[], n_type=n_type)

        pair_params, pair_params_cond, pair_cond = self._get_pair_params(cutoff)
        model = PolymlpModelParams(
            cutoff,
            model_type,
            max_p,
            max_l,
            feature_type=feature_type,
            gtinv=gtinv_params,
            pair_type="gaussian",
            pair_conditional=pair_cond,
            pair_params=pair_params,
            pair_params_conditional=pair_params_cond,
        )

        return model

    def _get_pair_params(self, cutoff):

        params1 = self.parser.get_sequence("gaussian_params1", default=(1.0, 1.0, 1))
        params2 = self.parser.get_sequence(
            "gaussian_params2",
            default=(0.0, cutoff - 1.0, 7),
        )
        pair_params = list(itertools.product(params1, params2))
        pair_params.append((0.0, 0.0))

        distance = self.parser.distance
        cond = False if len(distance) == 0 else True

        element_pairs = itertools.combinations_with_replacement(self._elements, 2)
        pair_params_indices = dict()
        for ele_pair in element_pairs:
            key = (self._atomtypes[ele_pair[0]], self._atomtypes[ele_pair[1]])
            if ele_pair not in distance:
                pair_params_indices[key] = list(range(len(pair_params)))
            else:
                match = [len(pair_params) - 1]
                for dis in distance[ele_pair]:
                    for i, p in enumerate(pair_params[:-1]):
                        if dis < p[1] + 1 / p[0] and dis > p[1] - 1 / p[0]:
                            match.append(i)
                pair_params_indices[key] = sorted(set(match))

        return pair_params, pair_params_indices, cond

    def _get_atomic_energy(self, n_type: int):

        d_atom_e = [0.0 for i in range(n_type)]
        atom_e = self.parser.get_params(
            "atomic_energy",
            size=n_type,
            default=d_atom_e,
            dtype=float,
            return_array=True,
        )
        return tuple(atom_e)

    def _get_regression_params(self):

        method = "ridge"
        d_alpha = [-3, 1, 5]
        alpha = self.parser.get_sequence("reg_alpha_params", default=d_alpha)
        return method, tuple(alpha)

    def _get_dataset(
        self,
        dataset_type: Literal["vasp", "phono3py"],
        multiple_datasets: bool = False,
        prefix: str = "/",
    ):
        if dataset_type == "vasp":
            if multiple_datasets:
                return self._get_multiple_vasprun_sets(prefix=prefix)
            else:
                return self._get_single_vasprun_set(prefix=prefix)
        elif dataset_type == "phono3py":
            return self._get_phono3py_set(prefix=prefix)

    def _get_single_vasprun_set(self, prefix=None):

        train = self.parser.get_params("train_data", default=None)
        test = self.parser.get_params("test_data", default=None)

        if prefix is None:
            dft_train = sorted(glob.glob(train))
            dft_test = sorted(glob.glob(test))
        else:
            dft_train = sorted(glob.glob(prefix + "/" + train))
            dft_test = sorted(glob.glob(prefix + "/" + test))
        return dft_train, dft_test

    def _get_multiple_vasprun_sets(self, prefix=None):

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

        if self.include_force == False:
            for params in train:
                params[1] = "False"
            for params in test:
                params[1] = "False"

        dft_train, dft_test = dict(), dict()
        for params in train:
            set_id = params[0]
            dft_train[set_id] = dict()
            if prefix is None:
                dft_train[set_id]["vaspruns"] = sorted(glob.glob(set_id))
            else:
                dft_train[set_id]["vaspruns"] = sorted(glob.glob(prefix + "/" + set_id))
            dft_train[set_id]["include_force"] = strtobool(params[1])
            dft_train[set_id]["weight"] = float(params[2])
        for params in test:
            set_id = params[0]
            dft_test[set_id] = dict()
            if prefix is None:
                dft_test[set_id]["vaspruns"] = sorted(glob.glob(set_id))
            else:
                dft_test[set_id]["vaspruns"] = sorted(glob.glob(prefix + "/" + set_id))
            dft_test[set_id]["include_force"] = strtobool(params[1])
            dft_test[set_id]["weight"] = float(params[2])
        return dft_train, dft_test

    def _get_phono3py_set(self, prefix=None):
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

        dft_train, dft_test = dict(), dict()
        if prefix is None:
            dft_train["phono3py_yaml"] = train[0]
            dft_test["phono3py_yaml"] = test[0]
        else:
            dft_train["phono3py_yaml"] = prefix + "/" + train[0]
            dft_test["phono3py_yaml"] = prefix + "/" + test[0]

        if len(train) == 2 or len(train) == 4:
            if prefix is None:
                dft_train["energy"] = train[1]
                dft_test["energy"] = test[1]
            else:
                dft_train["energy"] = prefix + "/" + train[1]
                dft_test["energy"] = prefix + "/" + test[1]

        if len(train) > 2:
            if phono3py_sample == "sequence":
                dft_train["indices"] = np.arange(int(train[-2]), int(train[-1]))
            elif phono3py_sample == "random":
                dft_train["indices"] = np.random.choice(
                    int(train[-2]), size=int(train[-1])
                )
        else:
            dft_train["indices"] = None

        if len(test) > 2:
            dft_test["indices"] = np.arange(int(test[-2]), int(test[-1]))
        else:
            dft_test["indices"] = None
        return dft_train, dft_test

    @property
    def params(self) -> PolymlpModelParams:
        """Return parameters for developing polymlp."""
        return self._params
