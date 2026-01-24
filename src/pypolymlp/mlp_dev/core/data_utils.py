"""Class for keeping X, y, X.T @ X and X.T @ y."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PolymlpDataXY:
    """Dataclass for keeping X, y, X.T @ X and X.T @ y.

    Parameters
    ----------
    x: Predictor matrix, shape=(total_n_data, n_features)
    y: Observation vector, shape=(total_n_data)
    xtx: Matrix x.T @ x.
    xty: Vector x.T @ y.
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

    def clear_data(self):
        """Clear large data."""
        del self.x, self.y, self.xtx, self.xty
        del self.weights, self.xe_sum, self.xe_sq_sum

    def slices(self, n_samples: int, total_n_atoms: np.ndarray):
        """Return slices for selected data."""
        if self.x is None:
            raise RuntimeError("Data X is not found.")

        ids = list(range(n_samples))

        first_id = self.first_indices[0][2]
        ids_stress = range(first_id, first_id + n_samples * 6)
        ids.extend(ids_stress)

        first_id = self.first_indices[0][1]
        n_forces = sum(total_n_atoms[:n_samples]) * 3
        ids_force = range(first_id, first_id + n_forces)
        ids.extend(ids_force)
        ids = np.array(ids)
        return self.x[ids], self.y[ids]


#     def apply_scales(
#         self,
#         scales: Optional[np.ndarray] = None,
#         include_force: bool = True,
#     ):
#         """Apply scales to X."""
#         if self.x is None:
#             raise RuntimeError("No data X found.")
#
#         if scales is None:
#             ne, nf, ns = self.n_data
#             scales = np.std(self.x[:ne], axis=0)
#         scales, zero_ids = round_scales(scales, include_force=include_force)
#
#         self.x[:, zero_ids] = 0.0
#         self.x /= scales
#         self.scales = scales
#         return self
#
#     def apply_weights(
#         self,
#         datasets: DatasetList,
#         min_energy: Optional[float] = None,
#         weight_stress: float = 0.1,
#     ):
#         """Apply weights to X and y."""
#         if self.x is None:
#             raise RuntimeError("No data X found.")
#         x = self.x
#         n_data = x.shape[0]
#         y = np.zeros(n_data)
#         w = np.ones(n_data)
#
#         if min_energy is None:
#             min_energy = get_min_energy(datasets)
#         self.min_energy = min_energy
#
#         for data, indices in zip(datasets, self.first_indices):
#             x, y, w = apply_weights(
#                 x,
#                 y,
#                 w,
#                 data,
#                 indices,
#                 weight_stress=weight_stress,
#                 min_e=min_energy,
#             )
#
#         self.x = x
#         self.y = y
#         self.weight = w
#         return self
