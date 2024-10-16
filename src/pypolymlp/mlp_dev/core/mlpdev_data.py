"""Class of preparing data structures for developing polymlp."""

import copy
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_dev.core.features_attr import (
    get_num_features,
    write_polymlp_params_yaml,
)
from pypolymlp.mlp_dev.core.parser_datasets import ParserDatasets


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


def set_unique_types(
    multiple_params: list[PolymlpParams],
    common_params: PolymlpParams,
):
    """Set type indices for hybrid models."""
    n_type = common_params.n_type
    elements = common_params.elements
    for single in multiple_params:
        single.elements = sorted(single.elements, key=lambda x: elements.index(x))

    for single in multiple_params:
        if single.n_type == n_type:
            single.type_full = True
            single.type_indices = list(range(n_type))
        else:
            single.type_full = False
            single.type_indices = [elements.index(ele) for ele in single.elements]
    return multiple_params


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
        prefix: str = None,
        verbose: bool = True,
    ):
        """Parse input files for developing polymlp."""
        if isinstance(infiles, list) == False:
            p = ParamsParser(infiles, multiple_datasets=True, prefix=prefix)
            self._params = p.params
            priority_infile = infiles
        else:
            priority_infile = infiles[0]
            if len(infiles) == 1:
                p = ParamsParser(priority_infile, multiple_datasets=True, prefix=prefix)
                self._params = p.params
            else:
                self._hybrid_params = [
                    ParamsParser(infile, multiple_datasets=True, prefix=prefix).params
                    for infile in infiles
                ]
                self._params = set_common_params(self._hybrid_params)
                self._hybrid_params = set_unique_types(
                    self._hybrid_params, self._params
                )
                self._hybrid = True

        if verbose:
            self.print_params(infile=priority_infile)
        return self

    def parse_datasets(self):
        """Parse DFT datasets."""
        if self._params is None:
            raise ValueError("PolymlpParams object is required.")

        parser = ParserDatasets(self._params)
        self._train = parser.train
        self._test = parser.test

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
