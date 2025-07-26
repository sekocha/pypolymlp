"""API Class for developing polymlp."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams

# from pypolymlp.core.parser_polymlp_params import parse_parameter_files
# from pypolymlp.mlp_dev.core.parser_datasets import ParserDatasets
from pypolymlp.mlp_dev.core.data import PolymlpDataXY, calc_xtx_xty, calc_xy
from pypolymlp.mlp_dev.core.utils import set_params

# from pypolymlp.mlp_dev.core.features_attr import (
# write_polymlp_params_yaml,
# )

# (
#     self._params,
#     self._hybrid_params,
#     self._hybrid,
#     self._priority_infile,
# ) = parse_parameter_files(infiles, prefix=prefix)
#
# parser = ParserDatasets(params, train_ratio=0.9)
# parser.train, parser.test


# def fit_standard(
#     params: Union[PolymlpParams, list[PolymlpParams]],
#     train: list[PolymlpDataDFT],
#     test: list[PolymlpDataDFT],
# ):
#
#     polymlp = PolymlpDev(params)
#     train_xy = polymlp.calc_xy(train)
#     test_xy = polymlp.calc_xy(
#         test,
#         scales=train_xy.scales,
#         min_energy=train_xy.min_energy,
#     )
#


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
