"""API Class for developing polymlp."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.accuracy import PolymlpEvalAccuracy
from pypolymlp.mlp_dev.core.data import PolymlpDataXY, calc_xtx_xty, calc_xy
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP
from pypolymlp.mlp_dev.core.features_attr import get_features_attr, get_num_features
from pypolymlp.mlp_dev.core.utils import check_memory_size_in_regression, set_params
from pypolymlp.mlp_dev.core.utils_model_selection import (
    compute_rmse,
    get_best_model,
    print_log,
)


class PolymlpDevCore:
    """API Class for developing polymlp."""

    def __init__(
        self,
        params: Union[PolymlpParams, list[PolymlpParams]],
        use_gradient: bool = False,
        verbose: bool = False,
    ):
        """Init method.."""
        self.params = params
        self._verbose = verbose
        self._use_gradient = use_gradient
        self._n_features = None

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
            use_gradient=self._use_gradient,
            verbose=self._verbose,
        )
        return data_xy

    def get_features_attr(self, element_swap: bool = False):
        """Return feature attributes."""
        features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(
            self._params,
            element_swap=element_swap,
        )
        return (features_attr, polynomial_attr, atomtype_pair_dict)

    def check_memory_size_in_regression(self):
        """Estimate memory size in regression."""
        return check_memory_size_in_regression(
            self.n_features,
            use_gradient=self._use_gradient,
            verbose=self._verbose,
        )

    def compute_rmse(
        self,
        coefs_array: np.ndarray,
        data_xy: Optional[PolymlpDataXY] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        check_singular: bool = False,
    ):
        """Compute RMSE for model coefficients."""
        return compute_rmse(
            coefs_array,
            data_xy=data_xy,
            x=x,
            y=y,
            check_singular=check_singular,
        )

    def get_best_model(
        self,
        coefs: np.ndarray,
        scales: np.ndarray,
        rmse_train: np.ndarray,
        rmse_test: np.ndarray,
        cumulative_n_features: Optional[tuple] = None,
    ):
        """Return best polymlp model."""
        return get_best_model(
            coefs,
            scales,
            self._common_params.alphas,
            rmse_train,
            rmse_test,
            self._params,
            cumulative_n_features,
        )

    def print_model_selection_log(self, rmse_train: np.ndarray, rmse_test: np.ndarray):
        """Print log from model selection."""
        print_log(rmse_train, rmse_test, self._common_params.alphas)

    @property
    def n_features(self):
        """Return number of features."""
        if self._n_features is None:
            self._n_features = get_num_features(self._params)
        return self._n_features

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


def eval_accuracy(
    mlp_model: PolymlpDataMLP,
    dft: list[PolymlpDataDFT],
    stress_unit: Literal["eV", "GPa"] = "eV",
    log_energy: bool = True,
    log_force: bool = False,
    log_stress: bool = False,
    path_output: str = "./",
    tag: str = "train",
    verbose: bool = False,
):
    """Evaluate accuracy."""
    acc = PolymlpEvalAccuracy(mlp_model, verbose=verbose)
    error = acc.compute_error(
        dft,
        stress_unit=stress_unit,
        log_energy=log_energy,
        log_force=log_force,
        log_stress=log_stress,
        path_output=path_output,
        tag=tag,
    )
    return error
