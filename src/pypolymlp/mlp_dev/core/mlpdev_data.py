"""Class of preparing data structures for developing polymlp."""

import copy
from typing import Self, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.interface_vasp import parse_vaspruns
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_dev.core.features_attr import (
    get_num_features,
    write_polymlp_params_yaml,
)


def get_variable_with_max_length(
    multiple_params: list[PolymlpParams], key: str
) -> list:

    array = []
    for single in multiple_params:
        single_dict = single.as_dict()
        if len(single_dict[key]) > len(array):
            array = single_dict[key]
    return array


def set_common_params(multiple_params: list[PolymlpParams]) -> PolymlpParams:
    """Set common parameters of multiple PolymlpParams."""
    keys = set()
    for single in multiple_params:
        for k in single.as_dict().keys():
            keys.add(k)

    common_params = copy.copy(multiple_params[0])
    n_type = max([single.n_type for single in multiple_params])
    elements = get_variable_with_max_length(multiple_params, "elements")
    atom_e = get_variable_with_max_length(multiple_params, "atomic_energy")

    bool_element_order = [
        single.element_order for single in multiple_params
    ] is not None
    element_order = elements if bool_element_order else None

    common_params.n_type = n_type
    common_params.elements = elements
    common_params.element_order = element_order
    common_params.atomic_energy = atom_e
    return common_params


class PolymlpDevData:
    """Class of preparing data structures for developing polymlp."""

    def __init__(self):

        self._params = None
        self._hybrid_params = None

        self._multiple_datasets = True
        self._hybrid = False

        self._train = None
        self._test = None
        self._n_features = None

    def parse_infiles(
        self,
        infiles: Union[str, list[str]],
        verbose: bool = True,
    ) -> Self:
        """Parse input files for developing polymlp."""
        if isinstance(infiles, list) is False:
            p = ParamsParser(infiles, multiple_datasets=True)
            self._params = p.params
            priority_infile = infiles
        else:
            priority_infile = infiles[0]
            if len(infiles) == 1:
                p = ParamsParser(priority_infile, multiple_datasets=True)
                self._params = p.params
            else:
                self._hybrid_params = [
                    ParamsParser(infile, multiple_datasets=True).params
                    for infile in infiles
                ]
                self._params = set_common_params(self._hybrid_params)
                self._hybrid = True

        if verbose:
            self.print_params(infile=priority_infile)
        return self

    def parse_datasets(self):
        if self._params.dataset_type == "vasp":
            if isinstance(self._params.dft_train, list):
                raise ValueError(
                    "Multiple_datasets format is needed for using vasprun files."
                )
            else:
                self.parse_multiple_datasets()
        elif self._params.dataset_type == "phono3py":
            self._parse_single_dataset()
            self._params.dft_train = {"train1": self._params.dft_train}
            self._params.dft_test = {"test1": self._params.dft_test}
            self._train.name = "train1"
            self._test.name = "test1"
            self._train = [self._train]
            self._test = [self._test]
            self._multiple_datasets = True
        else:
            raise ValueError("Only dataset types of vasp and phono3py are available.")

    def _parse_single_dataset(self) -> Self:

        if self._params is None:
            raise ValueError("PolymlpParams object is needed.")

        element_order = self._params.element_order
        dft_train = self._params.dft_train
        dft_test = self._params.dft_test
        if self._params.dataset_type == "vasp":
            self._train = parse_vaspruns(dft_train, element_order=element_order)
            self._test = parse_vaspruns(dft_test, element_order=element_order)
        elif self._params.dataset_type == "phono3py":
            """TODO"""
            from pypolymlp.core.interface_phono3py_ver3 import parse_phono3py_yaml

            self._train = parse_phono3py_yaml(
                dft_train["phono3py_yaml"],
                dft_train["energy"],
                element_order=element_order,
                select_ids=dft_train["indices"],
                use_phonon_dataset=False,
            )
            self._test = parse_phono3py_yaml(
                dft_test["phono3py_yaml"],
                dft_test["energy"],
                element_order=element_order,
                select_ids=dft_test["indices"],
                use_phonon_dataset=False,
            )

        self._train.name = "train_single"
        self._train.include_force = self._params.include_force
        self._train.apply_atomic_energy(self._params.atomic_energy)

        self._test.name = "test_single"
        self._test.include_force = self._params.include_force
        self._test.apply_atomic_energy(self._params.atomic_energy)
        return self

    def parse_multiple_datasets(self):

        if self._params is None:
            raise ValueError("PolymlpParams object is needed.")

        if self._params.dataset_type == "vasp":
            element_order = self._params.element_order
            self._train = []
            for name, dict1 in self._params.dft_train.items():
                dft = parse_vaspruns(dict1["vaspruns"], element_order=element_order)
                dft.apply_atomic_energy(self._params.atomic_energy)
                dft.name = name
                dft.include_force = dict1["include_force"]
                dft.weight = dict1["weight"]
                self._train.append(dft)

            self._test = []
            for name, dict1 in self._params.dft_test.items():
                dft = parse_vaspruns(dict1["vaspruns"], element_order=element_order)
                dft.apply_atomic_energy(self._params.atomic_energy)
                dft.name = name
                dft.include_force = dict1["include_force"]
                dft.weight = dict1["weight"]
                self._test.append(dft)
        else:
            raise KeyError("Only dataset_type = vasp is available.")

        self._multiple_datasets = True
        return self

    def print_params(self, infile=None):
        """Print parameters."""
        if infile is not None:
            print("priority_input:", infile, flush=True)

        params = self.common_params
        print("parameters:", flush=True)
        print("  n_types:       ", params.n_type, flush=True)
        print("  elements:      ", params.elements, flush=True)
        print("  element_order: ", params.element_order, flush=True)
        print("  atomic_energy: ", params.atomic_energy, flush=True)
        print("  include_force: ", bool(params.include_force), flush=True)
        print("  include_stress:", bool(params.include_stress), flush=True)

        if self.is_multiple_datasets:
            print("  train_data:", flush=True)
            for v in params.dft_train:
                print("  -", v, flush=True)
            print("  test_data:", flush=True)
            for v in params.dft_test:
                print("  -", v, flush=True)
        else:
            if params.dataset_type == "phono3py":
                print("  train_data:", flush=True)
                print("  -", params.dft_train["phono3py_yaml"], flush=True)
                print("  test_data:", flush=True)
                print("  -", params.dft_test["phono3py_yaml"], flush=True)
            else:
                pass

    def write_polymlp_params_yaml(self, filename="polymlp_params.yaml"):
        """Write polymlp_params.yaml"""
        if not self.is_hybrid:
            self._n_features = write_polymlp_params_yaml(self.params, filename=filename)
        else:
            self._n_features = 0
            for i, params in enumerate(self.params):
                filename = "polymlp_params" + str(i + 1) + ".yaml"
                self._n_features += write_polymlp_params_yaml(params, filename=filename)

    @property
    def params(self) -> Union[PolymlpParams, list[PolymlpParams]]:
        if self._hybrid:
            return self._hybrid_params
        return self._params

    @property
    def common_params(self) -> PolymlpParams:
        return self._params

    @property
    def hybrid_params(self) -> list[PolymlpParams]:
        return self._hybrid_params

    @params.setter
    def params(self, params: PolymlpParams):
        if isinstance(params, list):
            if len(params) > 1:
                self.hybrid_params = params
            else:
                self._params = params[0]
                self._hybrid = False
        else:
            self._params = params
            self._hybrid = False

    @hybrid_params.setter
    def hybrid_params(self, params: list[PolymlpParams]):
        self._hybrid_params = params
        self._params = set_common_params(params)
        self._hybrid = True

    @property
    def train(self) -> Union[PolymlpDataDFT, dict[PolymlpDataDFT]]:
        return self._train

    @property
    def test(self) -> Union[PolymlpDataDFT, dict[PolymlpDataDFT]]:
        return self._test

    @property
    def n_features(self) -> int:
        if not self.is_hybrid:
            self._n_features = get_num_features(self.params)
        else:
            self._n_features = 0
            for i, params in enumerate(self.params):
                self._n_features += get_num_features(params)
        return self._n_features

    @train.setter
    def train(self, data: Union[PolymlpDataDFT, dict[PolymlpDataDFT]]):
        self._train = data
        if isinstance(data, PolymlpDataDFT):
            self._multiple_datasets = False
        else:
            self._multiple_datasets = True

    @test.setter
    def test(self, data: Union[PolymlpDataDFT, dict[PolymlpDataDFT]]):
        self._test = data
        if isinstance(data, PolymlpDataDFT):
            self._multiple_datasets = False
        else:
            self._multiple_datasets = True

    @property
    def is_multiple_datasets(self) -> bool:
        return self._multiple_datasets

    @property
    def is_hybrid(self) -> bool:
        return self._hybrid

    @property
    def min_energy(self) -> float:
        if self._multiple_datasets:
            min_e = 1e10
            for dft in self._train:
                e_per_atom = dft.energies / dft.total_n_atoms
                min_e_trial = np.min(e_per_atom)
                if min_e_trial < min_e:
                    min_e = min_e_trial
        else:
            dft = self._train
            e_per_atom = dft.energies / dft.total_n_atoms
            min_e = np.min(e_per_atom)
        return min_e
