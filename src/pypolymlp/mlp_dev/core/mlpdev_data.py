"""Class of preparing data structures for developing polymlp."""

from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.parser_datasets import ParserDatasets
from pypolymlp.core.parser_polymlp_params import (
    parse_parameter_files,
    set_common_params,
)
from pypolymlp.mlp_dev.core.features_attr import (
    get_num_features,
    write_polymlp_params_yaml,
)


class PolymlpDevData:
    """Class of preparing data structures for developing polymlp."""

    def __init__(self):
        """Init method.."""
        self._params = None
        self._hybrid_params = None
        self._priority_infile = None
        self._hybrid = False

        self._train = None
        self._test = None
        self._multiple_datasets = True
        self._n_features = None

    def parse_infiles(self, infiles: Union[str, list[str]], prefix: str = None):
        """Parse input files for developing polymlp."""
        (
            self._params,
            self._hybrid_params,
            self._hybrid,
            self._priority_infile,
        ) = parse_parameter_files(infiles, prefix=prefix)
        return self

    def parse_datasets(self, train_ratio: float = 0.9):
        """Parse DFT datasets."""
        if self._params is None:
            raise ValueError("PolymlpParams object is required.")

        parser = ParserDatasets(self._params, train_ratio=train_ratio)
        self.set_datasets(parser.train, parser.test)
        return self

    def set_datasets(
        self,
        train: Union[PolymlpDataDFT, list[PolymlpDataDFT]],
        test: Union[PolymlpDataDFT, list[PolymlpDataDFT]],
    ):
        """Set DFT datasets."""
        self._train = train
        self._test = test
        return self

    def print_params(self):
        """Print parameters."""
        if self._hybrid:
            print("priority_input:", self._priority_infile, flush=True)

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
                print("  -", v.location, flush=True)
            print("  test_data:", flush=True)
            for v in params.dft_test:
                print("  -", v.location, flush=True)
        else:
            if params.dataset_type == "phono3py":
                print("  train_data:", flush=True)
                print("  -", params.dft_train["phono3py_yaml"], flush=True)
                print("  test_data:", flush=True)
                print("  -", params.dft_test["phono3py_yaml"], flush=True)
            else:
                pass

        if isinstance(self.params, PolymlpParams):
            params = [self.params]
        else:
            params = self.params
        for i, p in enumerate(params):
            print("model_" + str(i + 1) + ":", flush=True)
            print("  cutoff:      ", p.model.cutoff, flush=True)
            print("  model_type:  ", p.model.model_type, flush=True)
            print("  max_p:       ", p.model.max_p, flush=True)
            print("  n_gaussians: ", len(p.model.pair_params), flush=True)
            print("  feature_type:", p.model.feature_type, flush=True)
            if p.model.feature_type == "gtinv":
                orders = [i for i in range(2, p.model.gtinv.order + 1)]
                print("  max_l:       ", p.model.gtinv.max_l, end=" ", flush=True)
                print("for order =", orders, flush=True)

    def write_polymlp_params_yaml(self, filename="polymlp_params.yaml"):
        """Write polymlp_params.yaml"""
        np.set_printoptions(legacy="1.21")
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
    def train(self) -> Union[PolymlpDataDFT, list[PolymlpDataDFT]]:
        return self._train

    @property
    def test(self) -> Union[PolymlpDataDFT, list[PolymlpDataDFT]]:
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
        """Calculate minimum of DFT energies."""
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
