"""API Class for developing polymlp."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.parser_datasets import ParserDatasets
from pypolymlp.core.parser_polymlp_params import parse_parameter_files
from pypolymlp.mlp_dev.core.data import PolymlpDataXY, calc_xtx_xty, calc_xy
from pypolymlp.mlp_dev.core.utils import set_params

# from pypolymlp.mlp_dev.core.features_attr import (
# write_polymlp_params_yaml,
# )


class PolymlpDev:
    """API Class for developing polymlp."""

    def __init__(
        self,
        params: Union[PolymlpParams, list[PolymlpParams]],
        verbose: bool = False,
    ):
        """Init method.."""
        self.params = params
        self._verbose = verbose

    def calc_xy(
        self,
        dft: list[PolymlpDataDFT],
        element_swap: bool = False,
        scales: Optional[np.ndarray] = None,
        min_energy: Optional[float] = None,
        weight_stress: float = 0.1,
    ) -> PolymlpDataXY:
        """Calculate X and y data."""
        self._is_list(dft)
        data_xy = calc_xy(
            self.params,
            self.common_params,
            dft,
            element_swap=element_swap,
            scales=scales,
            min_energy=min_energy,
            weight_stress=weight_stress,
            verbose=self._verbose,
        )
        return data_xy

    def calc_xtx_xty(
        self,
        dft: list[PolymlpDataDFT],
        element_swap: bool = False,
        scales: Optional[np.ndarray] = None,
        min_energy: Optional[float] = None,
        weight_stress: float = 0.1,
        batch_size: Optional[int] = None,
    ) -> PolymlpDataXY:
        """Calculate X.T @ X and X.T @ y data."""
        self._is_list(dft)
        data_xy = calc_xtx_xty(
            self.params,
            self.common_params,
            dft,
            element_swap=element_swap,
            scales=scales,
            min_energy=min_energy,
            weight_stress=weight_stress,
            batch_size=batch_size,
            verbose=self._verbose,
        )
        return data_xy

    @property
    def params(self) -> Union[PolymlpParams, list[PolymlpParams]]:
        """Return polymlp parameters."""
        return self._params

    @property
    def common_params(self) -> PolymlpParams:
        """Return common parameters in hybrid polymlp."""
        return self._common_params

    @params.setter
    def params(self, params: Union[PolymlpParams, list[PolymlpParams]]):
        """Set parameters."""
        self._params, self._common_params, _ = set_params(params)
        self._hybrid = True if isinstance(self._params, list) else False

    @property
    def is_hybrid(self) -> bool:
        """Return whether hybrid model is used."""
        return self._hybrid

    def _is_list(self, dft: list[PolymlpDataDFT]):
        """Return whether DFT data is given by list or not."""
        if not isinstance(dft, (list, tuple, np.ndarray)):
            raise RuntimeError("DFT data must be given in list.")


#    def write_polymlp_params_yaml(self, filename="polymlp_params.yaml"):
#        """Write polymlp_params.yaml"""
#        np.set_printoptions(legacy="1.21")
#        self._n_features = write_polymlp_params_yaml(self.params, filename=filename)


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


#     def print_params(self):
#         """Print parameters."""
#         if self._hybrid:
#             print("priority_input:", self._priority_infile, flush=True)
#
#         params = self.common_params
#         print("parameters:", flush=True)
#         print("  n_types:       ", params.n_type, flush=True)
#         print("  elements:      ", params.elements, flush=True)
#         print("  element_order: ", params.element_order, flush=True)
#         print("  atomic_energy: ", params.atomic_energy, flush=True)
#         print("  include_force: ", bool(params.include_force), flush=True)
#         print("  include_stress:", bool(params.include_stress), flush=True)
#
#         if self.is_multiple_datasets:
#             print("  train_data:", flush=True)
#             for v in params.dft_train:
#                 print("  -", v.location, flush=True)
#             print("  test_data:", flush=True)
#             for v in params.dft_test:
#                 print("  -", v.location, flush=True)
#         else:
#             if params.dataset_type == "phono3py":
#                 print("  train_data:", flush=True)
#                 print("  -", params.dft_train["phono3py_yaml"], flush=True)
#                 print("  test_data:", flush=True)
#                 print("  -", params.dft_test["phono3py_yaml"], flush=True)
#             else:
#                 pass
#
#         if isinstance(self.params, PolymlpParams):
#             params = [self.params]
#         else:
#             params = self.params
#         for i, p in enumerate(params):
#             print("model_" + str(i + 1) + ":", flush=True)
#             print("  cutoff:      ", p.model.cutoff, flush=True)
#             print("  model_type:  ", p.model.model_type, flush=True)
#             print("  max_p:       ", p.model.max_p, flush=True)
#             print("  n_gaussians: ", len(p.model.pair_params), flush=True)
#             print("  feature_type:", p.model.feature_type, flush=True)
#             if p.model.feature_type == "gtinv":
#                 orders = [i for i in range(2, p.model.gtinv.order + 1)]
#                 print("  max_l:       ", p.model.gtinv.max_l, end=" ", flush=True)
#                 print("for order =", orders, flush=True)
#
#    def write_polymlp_params_yaml(self, filename="polymlp_params.yaml"):
#        """Write polymlp_params.yaml"""
#        np.set_printoptions(legacy="1.21")
#        self._n_features = write_polymlp_params_yaml(self.params, filename=filename)
