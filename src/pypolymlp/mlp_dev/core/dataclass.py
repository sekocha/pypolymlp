"""Dataclasses used for regression."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.utils import get_min_energy, round_scales
from pypolymlp.core.utils_weights import apply_weights


@dataclass
class PolymlpDataXY:
    """Dataclass of X, y, and related properties used for regression.

    Parameters
    ----------
    x: Predictor matrix, shape=(total_n_data, n_features)
    y: Observation vector, shape=(total_n_data)
    xtx: x.T @ x
    xty: x.T @ y
    scales: Scales of x, shape=(n_features)
    weights: Weights for data, shape=(total_n_data)
    n_data: Number of data (energy, force, stress)
    """

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    xtx: Optional[np.ndarray] = None
    xty: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    min_energy: Optional[float] = None

    n_data: Optional[tuple[int, int, int]] = None
    first_indices: Optional[list[tuple[int, int, int]]] = None
    cumulative_n_features: Optional[int] = None
    xe_sum: Optional[np.ndarray] = None
    xe_sq_sum: Optional[np.ndarray] = None
    y_sq_norm: float = 0.0
    total_n_data: int = 0

    def apply_scales(
        self, scales: Optional[np.ndarray] = None, include_force: bool = True
    ):
        """Apply scales to X."""
        if self.x is None:
            raise RuntimeError("No data X found.")

        if scales is None:
            ne, nf, ns = self.n_data
            scales = np.std(self.x[:ne], axis=0)
        scales, zero_ids = round_scales(scales, include_force=include_force)

        self.x[:, zero_ids] = 0.0
        self.x /= scales
        self.scales = scales
        return self

    def apply_weights(
        self,
        common_params: PolymlpParams,
        dft_all: list[PolymlpDataDFT],
        min_energy: Optional[float] = None,
        weight_stress: float = 0.1,
    ):
        """Apply weights to X and y."""
        # TODO: common_params
        if self.x is None:
            raise RuntimeError("No data X found.")
        x = self.x
        n_data = x.shape[0]
        y = np.zeros(n_data)
        w = np.ones(n_data)

        if min_energy is None:
            min_energy = get_min_energy(dft_all)
        self.min_energy = min_energy

        for dft, indices in zip(dft_all, self.first_indices):
            x, y, w = apply_weights(
                x,
                y,
                w,
                dft,
                common_params,
                indices,
                weight_stress=weight_stress,
                min_e=min_energy,
            )

        self.x = x
        self.y = y
        self.weight = w
        return self


@dataclass
class PolymlpDataMLP:
    """Dataclass of regression results.

    Parameters
    ----------
    coeffs: MLP coefficients, shape=(n_features).
    scales: Scales of x, shape=(n_features).
    """

    coeffs: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    rmse: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    predictions_train: Optional[np.ndarray] = None
    predictions_test: Optional[np.ndarray] = None
    error_train: Optional[dict] = None
    error_test: Optional[dict] = None
