"""Dataclass for polymlp model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.io_polymlp import save_mlps
from pypolymlp.core.params import PolymlpParams


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

    params: Optional[PolymlpParams] = None
    cumulative_n_features: Optional[tuple] = None
    scaled_coeffs: Optional[np.ndarray] = None
    scaled_coeffs_flat: Optional[np.ndarray] = None

    def __post_init__(self):
        """Post init method."""
        if len(self.coeffs) != len(self.scales):
            raise RuntimeError("Found different lengths of coeffs and scales.")

        if self.cumulative_n_features is not None:
            n_features = self.cumulative_n_features[-1]
            if len(self.coeffs) != n_features:
                raise RuntimeError("Coefficient size not appropriate.")
            if len(self.scales) != n_features:
                raise RuntimeError("Scale size not appropriate.")

            coeffs_hybrid = self.hybrid_division(self.coeffs)
            scales_hybrid = self.hybrid_division(self.scales)
            self.scaled_coeffs = [c / s for c, s in zip(coeffs_hybrid, scales_hybrid)]
            self.scaled_coeffs_flat = []
            for c in self.scaled_coeffs:
                self.scaled_coeffs_flat.extend(c)
            self.scaled_coeffs_flat = np.array(self.scaled_coeffs_flat)
        else:
            self.scaled_coeffs = self.coeffs / self.scales
            self.scaled_coeffs_flat = self.scaled_coeffs

    def save_mlp(self, filename: str = "polymlp.yaml"):
        """Save polymlp.yaml files"""
        if self.params is None:
            raise RuntimeError("Parameters not found.")

        save_mlps(
            self.params,
            self.coeffs,
            self.scales,
            cumulative_n_features=self.cumulative_n_features,
            filename=filename,
        )
        return self

    def hybrid_division(self, target: np.ndarray):
        """Divide array for hybrid model."""
        cumulative = self.cumulative_n_features
        list_target = []
        for i, params in enumerate(self.params):
            begin = 0 if i == 0 else cumulative[i - 1]
            end = cumulative[i]
            list_target.append(np.array(target[begin:end]))
        return list_target
