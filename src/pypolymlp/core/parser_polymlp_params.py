"""Class of input parameter parser."""

import glob
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParams
from pypolymlp.core.parser_infile import InputParser
from pypolymlp.core.polymlp_params import (
    set_active_gaussian_params,
    set_gaussian_params,
    set_gtinv_params,
    set_regression_alphas,
)
from pypolymlp.core.utils import split_train_test, strtobool


class ParamsParser:
    """Class of input parameter parser."""

    def __init__(
        self,
        filename: str,
        multiple_datasets: bool = False,
        parse_vasprun_locations: bool = True,
        prefix: Optional[str] = None,
    ):
        """Init class.

        Parameters
        ----------
        filename: File of input parameters for single polymlp (e.g., polymlp.in).
        """
        self.parser = InputParser(filename)
        include_force, include_stress = self._set_force_tags()
        self.include_force = include_force

        elements, n_type, atomic_energy = self._set_element_properties()
        self._elements = elements
        rearrange = self.parser.get_params(
            "rearrange_by_elements",
            default=True,
            dtype=bool,
        )
        element_order = elements if rearrange else None
        alphas = self._get_regression_params()
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
            regression_alpha=alphas,
            include_force=include_force,
            include_stress=include_stress,
            dataset_type=dataset_type,
            element_order=element_order,
        )

        if dataset_type == "electron":
            self._params.temperature = self.parser.get_params(
                "temperature", default=300, dtype=float
            )
            self._params.electron_property = self.parser.get_params(
                "electron_property", default="free_energy", dtype=str
            )

    def _set_force_tags(self):
        """Set include_force and include_stress."""
        include_force = self.parser.get_params(
            "include_force",
            default=True,
            dtype=bool,
        )
        if include_force:
            include_stress = self.parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            include_stress = False
        return include_force, include_stress

    def _set_element_properties(self):
        """Set properties for identifying elements."""
        n_type = self.parser.get_params("n_type", default=1, dtype=int)
        elements = self.parser.get_params(
            "elements",
            size=n_type,
            default=None,
            required=True,
            dtype=str,
            return_array=True,
        )
        d_atom_e = [0.0 for i in range(n_type)]
        atom_e = self.parser.get_params(
            "atomic_energy",
            size=n_type,
            default=d_atom_e,
            dtype=float,
            return_array=True,
        )
        return elements, n_type, tuple(atom_e)

    def _get_regression_params(self):
        """Set regularization parameters in regression."""
        alpha_params = self.parser.get_params(
            "reg_alpha_params",
            size=3,
            default=(-3.0, 1.0, 5),
            dtype=float,
            return_array=True,
        )
        alphas = set_regression_alphas(alpha_params)
        return alphas

    def _get_gtinv_params(self, n_type: int, feature_type: Literal["gtinv", "pair"]):
        """Set parameters for group-theoretical invariants."""
        version = self.parser.get_params("gtinv_version", default=1, dtype=int)
        order = self.parser.get_params("gtinv_order", default=3, dtype=int)
        size = order - 1
        gtinv_maxl = self.parser.get_params(
            "gtinv_maxl",
            size=size,
            default=[2 for i in range(size)],
            dtype=int,
            return_array=True,
        )
        if len(gtinv_maxl) < size:
            size_gap = size - len(gtinv_maxl)
            gtinv_maxl.extend([2 for i in range(size_gap)])

        gtinv_params, max_l = set_gtinv_params(
            n_type,
            feature_type=feature_type,
            gtinv_order=order,
            gtinv_maxl=gtinv_maxl,
            gtinv_version=version,
        )
        return gtinv_params, max_l

    def _get_potential_model_params(self, n_type: int):
        """Set parameters for identifying potential model."""
        cutoff = self.parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self.parser.get_params("model_type", default=1, dtype=int)
        max_p = self.parser.get_params("max_p", default=1, dtype=int)
        feature_type = self.parser.get_params("feature_type", default="gtinv")

        gtinv_params, max_l = self._get_gtinv_params(n_type, feature_type)
        pair_params, pair_params_active, pair_cond = self._get_pair_params(cutoff)

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
            pair_params_conditional=pair_params_active,
        )
        return model

    def _get_pair_params(self, cutoff: float):
        """Set parameters for Gaussian radial functions."""
        params1 = self.parser.get_params(
            "gaussian_params1",
            size=3,
            default=(1.0, 1.0, 1),
            dtype=float,
            return_array=True,
        )
        params2 = self.parser.get_params(
            "gaussian_params2",
            size=3,
            default=(0.0, cutoff - 1.0, 7),
            dtype=float,
            return_array=True,
        )
        distance = self.parser.distance
        pair_params = set_gaussian_params(params1, params2)
        pair_params_active, cond = set_active_gaussian_params(
            pair_params,
            self._elements,
            distance,
        )
        return pair_params, pair_params_active, cond

    def _get_dataset(
        self,
        dataset_type: Literal["vasp", "phono3py", "sscha", "electron"],
        multiple_datasets: bool = False,
        prefix: Optional[str] = None,
    ):
        """Parse filenames in dataset."""
        if dataset_type == "vasp":
            if multiple_datasets:
                return self._get_multiple_vasprun_sets(prefix=prefix)
            return self._get_single_vasprun_set(prefix=prefix)
        elif dataset_type == "phono3py":
            return self._get_phono3py_set(prefix=prefix)
        elif dataset_type == "sscha":
            return self._get_sscha_set(prefix=prefix)
        elif dataset_type == "electron":
            return self._get_electron_set(prefix=prefix)
        else:
            raise KeyError("Given dataset_type is unavailable.")

    def _get_sscha_set(self, prefix: Optional[str] = None):
        """Parse sscha_results.yaml files in dataset."""
        data = self.parser.get_params("data", default=None)
        if prefix is not None:
            data = prefix + "/" + data

        data_all = sorted(glob.glob(data))
        dft_train, dft_test = split_train_test(data_all, train_ratio=0.9)
        return dft_train, dft_test

    def _get_electron_set(self, prefix: Optional[str] = None):
        """Parse electron.yaml files in dataset."""
        data = self.parser.get_params("data", default=None)
        if prefix is not None:
            data = prefix + "/" + data

        data_all = sorted(glob.glob(data))
        dft_train, dft_test = split_train_test(data_all, train_ratio=0.9)
        return dft_train, dft_test

    def _get_single_vasprun_set(self, prefix: Optional[str] = None):
        """Parse vasprun filenames in dataset."""
        train = self.parser.get_params("train_data", default=None)
        test = self.parser.get_params("test_data", default=None)

        if prefix is None:
            dft_train = sorted(glob.glob(train))
            dft_test = sorted(glob.glob(test))
        else:
            dft_train = sorted(glob.glob(prefix + "/" + train))
            dft_test = sorted(glob.glob(prefix + "/" + test))
        return dft_train, dft_test

    def _get_multiple_vasprun_sets(self, prefix: Optional[str] = None):
        """Parse vasprun filenames in multiple datasets."""
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
