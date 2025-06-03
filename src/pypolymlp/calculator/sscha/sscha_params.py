"""Utility functions for SSCHA."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


@dataclass
class SSCHAParameters:
    """Dataclass of sscha parameters.

    Parameters
    ----------
    unitcell: Unit cell.
    supercell_matrix: Supercell matrix. shape=(3, 3).
    pot: polymlp path.
    temperatures: Simulation temperatures.
    temp: Single simulation temperature.
    temp_min: Minimum temperature.
    temp_max: Maximum temperature.
    temp_step: Temperature interval.
    ascending_temp: Set simulation temperatures in ascending order.
    n_samples_init: Number of samples in first loop of SSCHA iterations.
    n_samples_final: Number of samples in second loop of SSCHA iterations.
    tol: Convergence tolerance for FCs.
    max_iter: Maximum number of iterations.
    mixing: Mixing parameter.
            FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
    mesh: q-point mesh for computing harmonic properties using effective FC2.
    init_fc_algorithm: Algorithm for generating initial FCs.
    init_fc_file: If algorithm = "file", coefficients are read from given fc2.hdf5.
    nac_params: Parameters for non-analytic correction in phonon calculations.
    """

    unitcell: PolymlpStructure
    supercell_matrix: np.ndarray
    supercell: Optional[np.ndarray] = None
    pot: Optional[str] = None
    temperatures: Optional[np.ndarray] = None
    temp: Optional[float] = None
    temp_min: float = 0
    temp_max: float = 2000
    temp_step: float = 50
    ascending_temp: bool = (False,)
    n_samples_init: Optional[int] = None
    n_samples_final: Optional[int] = None
    tol: float = 0.01
    max_iter: int = 30
    mixing: float = 0.5
    mesh: tuple = (10, 10, 10)
    init_fc_algorithm: Literal["harmonic", "file"] = "harmonic"
    init_fc_file: Optional[str] = None
    nac_params: Optional[dict] = None
    cutoff_radius: Optional[float] = None

    def __post_init__(self):
        """Post init method."""
        self._n_atom = len(self.unitcell.elements) * np.linalg.det(
            self.supercell_matrix
        )
        if self.temperatures is None:
            self.set_temperatures()
        # self.set_n_samples()

    def _round_temperature(self, temp: float):
        """Round a single temperature if possible."""
        if np.isclose(temp, round(temp)):
            return round(temp)
        return temp

    def set_temperatures(self):
        """Set simulation temperatures."""
        if self.temp is not None:
            self.temperatures = [self._round_temperature(self.temp)]
            return self.temperatures

        self.temp_min = self._round_temperature(self.temp_min)
        self.temp_max = self._round_temperature(self.temp_max)
        self.temp_step = self._round_temperature(self.temp_step)
        self.temperatures = np.arange(
            self.temp_min,
            self.temp_max + 1,
            self.temp_step,
        )
        if not self.ascending_temp:
            self.temperatures = self.temperatures[::-1]
        return self.temperatures

    def set_n_samples(self):
        """Set number of supercells."""
        if self.n_samples_init is None:
            self.n_samples_unit = round(6400 / self._n_atom)
            self.n_samples_init = 20 * self.n_samples_unit
        if self.n_samples_final is None:
            self.n_samples_unit = round(6400 / self._n_atom)
            self.n_samples_final = 60 * self.n_samples_unit
        return self.n_samples_init, self.n_samples_final

    def set_n_samples_from_basis(self, n_basis: int):
        """Set number of supercells using basis size."""
        coeff = 300 * (0.01 / self.tol) ** 2
        self.n_samples_init = round(coeff * n_basis / self._n_atom)
        self.n_samples_final = 3 * self.n_samples_init
        return self.n_samples_init, self.n_samples_final

    def print_params(self):
        """Print parameters in SSCHA."""
        print(" # SSCHA parameters", flush=True)
        print("  - supercell:    ", self.supercell_matrix[0], flush=True)
        print("                  ", self.supercell_matrix[1], flush=True)
        print("                  ", self.supercell_matrix[2], flush=True)
        print("  - temperatures: ", self.temperatures[0], flush=True)
        if len(self.temperatures) > 1:
            for t in self.temperatures[1:]:
                print("                  ", t, flush=True)

        if isinstance(self.pot, list):
            for p in self.pot:
                print("  - Polymlp:      ", os.path.abspath(p), flush=True)
        else:
            print("  - Polymlp:      ", os.path.abspath(self.pot), flush=True)

        print("  - FC tolerance:             ", self.tol, flush=True)
        print("  - max iter:                 ", self.max_iter, flush=True)
        print("  - mixing:                   ", self.mixing, flush=True)
        print("  - num samples (first loop): ", self.n_samples_init, flush=True)
        print("  - num samples (second loop):", self.n_samples_final, flush=True)
        print("  - q-mesh:                   ", self.mesh, flush=True)
        if self.nac_params is not None:
            print("  - NAC params:                True", flush=True)

    def print_unitcell(self):
        """Print unitcell."""
        print(" # unit cell", flush=True)
        print("  - elements:     ", self.unitcell.elements, flush=True)
        print("  - axis:         ", self.unitcell.axis.T[0], flush=True)
        print("                  ", self.unitcell.axis.T[1], flush=True)
        print("                  ", self.unitcell.axis.T[2], flush=True)
        print("  - positions:    ", self.unitcell.positions.T[0], flush=True)
        if self.unitcell.positions.shape[1] > 1:
            for pos in self.unitcell.positions.T[1:]:
                print("                  ", pos, flush=True)
