"""Dataclass for polymlp model."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.io_polymlp import save_mlp, save_mlps
from pypolymlp.core.io_polymlp_legacy import save_mlp_lammps, save_multiple_mlp_lammps


@dataclass
class PolymlpDataMLP:
    """Dataclass for polymlp model.

    Parameters
    ----------
    coeffs: MLP coefficients, shape=(n_features).
    scales: Scales of x, shape=(n_features).
    """

    coeffs: np.ndarray
    scales: np.ndarray
    rmse_train: Optional[float] = None
    rmse_test: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    error_train: Optional[dict] = None
    error_test: Optional[dict] = None

    params: Optional[Union[PolymlpParams, list[PolymlpParams]]] = None

    cumulative_n_features: Optional[tuple] = None
    hybrid: bool = False
    coeffs_hybrid: Optional[np.ndarray] = None
    scales_hybrid: Optional[np.ndarray] = None
    scaled_coeffs: Optional[np.ndarray] = None

    def __post_init__(self):
        """Post init method."""
        if self.cumulative_n_features is not None:
            self.hybrid = True

        if self.hybrid:
            self.coeffs_hybrid = self.hybrid_division(self.coeffs)
            self.scales_hybrid = self.hybrid_division(self.scales)
            self.scaled_coeffs = [
                c / s for c, s in zip(self.coeffs_hybrid, self.scales_hybrid)
            ]
        else:
            self.scaled_coeffs = self.coeffs / self.scales

    def save_mlp(self, filename: str = "polymlp.yaml"):
        """Save polymlp.yaml files"""
        if not self.hybrid:
            save_mlp(
                self.params,
                self.coeffs,
                self.scales,
                filename=filename,
            )
        else:
            save_mlps(
                self.params,
                self.cumulative_n_features,
                self.coeffs,
                self.scales,
                prefix=filename,
            )
        return self

    def save_mlp_lammps(self, filename: str = "polymlp.lammps"):
        """Save polymlp.lammps files"""
        if not self.hybride:
            save_mlp_lammps(
                self.params,
                self.coeffs,
                self.scales,
                filename=filename,
            )
        else:
            save_multiple_mlp_lammps(
                self.params,
                self.cumulative_n_features,
                self.coeffs,
                self.scales,
                prefix=filename,
            )
        return self

    def hybrid_division(self, target: np.ndarray):
        """Divide array for hybrid model."""
        cumulative = self.cumulative_n_features
        list_target = []
        for i, params in enumerate(self.params):
            if i == 0:
                begin, end = 0, cumulative[0]
            else:
                begin, end = cumulative[i - 1], cumulative[i]
            list_target.append(np.array(target[begin:end]))
        return list_target
