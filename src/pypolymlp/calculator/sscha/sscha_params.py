"""Utility functions for SSCHA parameters."""

import os
from typing import Literal, Optional, Sequence, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


class SSCHAParams:
    """Container for SSCHA parameters."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        supercell: Optional[PolymlpStructure] = None,
        pot: Optional[Union[str, Sequence[str]]] = None,
        temperatures: Optional[Union[Sequence[float], np.ndarray]] = None,
        temp: Optional[float] = None,
        temp_min: float = 0,
        temp_max: float = 2000,
        temp_step: float = 50,
        n_temp: Optional[int] = None,
        ascending_temp: bool = False,
        n_samples_init: Optional[int] = None,
        n_samples_final: Optional[int] = None,
        tol: float = 0.01,
        max_iter: int = 30,
        mixing: float = 0.5,
        mesh: tuple = (10, 10, 10),
        init_fc_algorithm: Literal["harmonic", "file"] = "harmonic",
        init_fc_file: Optional[str] = None,
        nac_params: Optional[dict] = None,
        cutoff_radius: Optional[float] = None,
    ):
        """Init method.

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
        n_temp: Number of temperatures.
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

        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        self._supercell = supercell
        self._pot = pot
        self._temperatures = (
            np.array(temperatures) if temperatures is not None else None
        )
        self._temp = temp
        self._temp_min = temp_min
        self._temp_max = temp_max
        self._temp_step = temp_step
        self._n_temp = n_temp
        self._ascending_temp = ascending_temp
        self._n_samples_init = n_samples_init
        self._n_samples_final = n_samples_final
        self._tol = tol
        self._max_iter = max_iter
        self._mixing = mixing
        self._mesh = mesh
        self._init_fc_algorithm = init_fc_algorithm
        self._init_fc_file = init_fc_file
        self._nac_params = nac_params
        self._cutoff_radius = cutoff_radius

        self._update_n_atom()
        if self._temperatures is None:
            self.set_temperatures()

        self._n_unitcells = int(round(np.linalg.det(supercell_matrix)))
        self._n_atom = len(unitcell.elements) * self._n_unitcells

    def _update_n_atom(self):
        """Recompute the internal `_n_atom` derived quantity."""
        try:
            det = float(np.linalg.det(self._supercell_matrix))
        except Exception:
            det = 1.0
        self._n_atom = len(self._unitcell.elements) * det

    def _round_temperature(self, temp: float):
        """Round a temperature to int when it is very close to an integer."""
        if np.isclose(temp, round(temp)):
            return int(round(temp))
        return float(temp)

    def set_temperatures(self):
        """Generate the temperatures array according to the configured options."""
        # If explicit single temperature provided, use it.
        if self._temp is not None:
            self._temperatures = np.array([self._round_temperature(self._temp)])
            return self._temperatures

        # Use Chebyshev nodes if requested.
        if self._n_temp is not None:
            if np.isclose(self._temp_max, self._temp_min):
                self._temperatures = np.array([self._round_temperature(self._temp_max)])
                return self._temperatures

            chebyshev_nodes = np.cos(np.linspace(np.pi, 0, self._n_temp))
            dt = self._temp_max - self._temp_min
            temps = dt * (chebyshev_nodes + 1) / 2 + self._temp_min
            temps = np.round(temps)
            temps = np.array([self._round_temperature(t) for t in temps])
            self._temperatures = temps
            return self._temperatures

        # Default: arithmetic progression
        self._temp_min = self._round_temperature(self._temp_min)
        self._temp_max = self._round_temperature(self._temp_max)
        self._temp_step = self._round_temperature(self._temp_step)
        temps = np.arange(self._temp_min, self._temp_max + 1, self._temp_step)
        if not self._ascending_temp:
            temps = temps[::-1]
        self._temperatures = temps
        return self._temperatures

    def set_n_samples(self):
        """Determine number of samples for SSCHA loops based on `_n_atom`.

        Uses the original heuristics:
        - n_samples_unit = round(6400 / _n_atom)
        - n_samples_init = 20 * n_samples_unit (if not provided)
        - n_samples_final = 60 * n_samples_unit (if not provided)
        """
        if self._n_atom is None or self._n_atom == 0:
            raise RuntimeError("Cannot determine _n_atom for sample sizing")

        if self._n_samples_init is None:
            n_samples_unit = round(6400 / self._n_atom)
            self._n_samples_init = 20 * n_samples_unit
        if self._n_samples_final is None:
            if n_samples_unit is None:
                n_samples_unit = round(6400 / self._n_atom)
            self._n_samples_final = 60 * n_samples_unit
        return self._n_samples_init, self._n_samples_final

    def set_n_samples_from_basis(self, n_basis: int):
        """Compute sample counts from basis size.

        coeff = 300 * (0.01 / tol) ** 2.5
        n_samples_init = round(coeff * n_basis / _n_atom)
        n_samples_final = 3 * n_samples_init
        """
        if self._n_atom is None or self._n_atom == 0:
            raise RuntimeError("Cannot determine _n_atom for sample sizing")
        coeff = 300 * (0.01 / self._tol) ** 2.5
        self._n_samples_init = round(coeff * n_basis / self._n_atom)
        self._n_samples_final = 3 * self._n_samples_init
        return self._n_samples_init, self._n_samples_final

    def print_params(self):
        """Print key SSCHA parameters for debugging or logging."""
        print(" # SSCHA parameters", flush=True)
        print("  - supercell:    ", self._supercell_matrix[0], flush=True)
        print("                  ", self._supercell_matrix[1], flush=True)
        print("                  ", self._supercell_matrix[2], flush=True)

        if self._temperatures is None or len(self._temperatures) == 0:
            print("  - temperatures: None", flush=True)
        else:
            print("  - temperatures: ", self._temperatures[0], flush=True)
            if len(self._temperatures) > 1:
                for t in self._temperatures[1:]:
                    print("                  ", t, flush=True)

        if isinstance(self._pot, (list, tuple)):
            for p in self._pot:
                print("  - Polymlp:      ", os.path.abspath(p), flush=True)
        elif self._pot is not None:
            print("  - Polymlp:      ", os.path.abspath(self._pot), flush=True)
        else:
            print("  - Polymlp:      None", flush=True)

        print("  - FC tolerance:             ", self._tol, flush=True)
        print("  - max iter:                 ", self._max_iter, flush=True)
        print("  - mixing:                   ", self._mixing, flush=True)
        print("  - num samples (first loop): ", self._n_samples_init, flush=True)
        print("  - num samples (second loop):", self._n_samples_final, flush=True)
        print("  - q-mesh:                   ", self._mesh, flush=True)
        if self._nac_params is not None:
            print("  - NAC params:                True", flush=True)

    def print_unitcell(self):
        """Print unitcell details (elements, lattice axes, positions)."""
        print(" # unit cell", flush=True)
        print("  - elements:     ", self._unitcell.elements, flush=True)
        print("  - axis:         ", self._unitcell.axis.T[0], flush=True)
        print("                  ", self._unitcell.axis.T[1], flush=True)
        print("                  ", self._unitcell.axis.T[2], flush=True)
        print("  - positions:    ", self._unitcell.positions.T[0], flush=True)
        if self._unitcell.positions.shape[1] > 1:
            for pos in self._unitcell.positions.T[1:]:
                print("                  ", pos, flush=True)

    @property
    def unitcell(self) -> PolymlpStructure:
        """Return the stored unitcell (PolymlpStructure-like object)."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, value: PolymlpStructure):
        """Set unitcell and update derived quantities.

        Minimal validation: ensure object exposes `elements` and `axis`.
        """
        if value is None:
            raise ValueError("unitcell cannot be None")
        if not hasattr(value, "elements") or not hasattr(value, "axis"):
            raise TypeError("unitcell must be a PolymlpStructure-like object")
        self._unitcell = value
        self._update_n_atom()

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return the 3x3 supercell matrix."""
        return self._supercell_matrix

    @supercell_matrix.setter
    def supercell_matrix(self, value: np.ndarray):
        """Set supercell_matrix and update derived quantities.

        Enforces shape (3,3).
        """
        if value is None:
            raise ValueError("supercell_matrix cannot be None")
        arr = np.array(value)
        if arr.shape != (3, 3):
            raise ValueError("supercell_matrix must have shape (3,3)")
        self._supercell_matrix = arr
        self._update_n_atom()

    @property
    def supercell(self) -> Optional[PolymlpStructure]:
        """Optional supercell array (may be None)."""
        return self._supercell

    @supercell.setter
    def supercell(self, value: Optional[PolymlpStructure]):
        """Set the supercell array; convert to numpy array if provided."""
        if value is None:
            self._supercell = None
            return
        self._supercell = value

    @property
    def pot(self) -> Optional[Union[str, Sequence[str]]]:
        """Path to polymlp potential(s) or None."""
        return self._pot

    @pot.setter
    def pot(self, value: Optional[Union[str, Sequence[str]]]):
        """Set potential path(s). Accepts a string or list/tuple of strings."""
        if value is None:
            self._pot = None
            return
        if isinstance(value, (str, list, tuple)):
            self._pot = value
        else:
            raise TypeError("pot must be a path string or list/tuple of path strings")

    @property
    def temperatures(self) -> Optional[np.ndarray]:
        """Array of temperatures used for simulations (or None)."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, value: Optional[Union[Sequence[float], np.ndarray]]):
        """Set temperatures; ensure numeric array and round near-integers."""
        if value is None:
            self._temperatures = None
            return
        arr = np.array(value)
        arr = np.array([self._round_temperature(float(t)) for t in arr])
        self._temperatures = arr

    @property
    def temp(self) -> Optional[float]:
        """Single temperature override (setting this will replace temperatures)."""
        return self._temp

    @temp.setter
    def temp(self, value: Optional[float]):
        """Set single temperature; apply it immediately to `temperatures`."""
        self._temp = None if value is None else float(value)
        if self._temp is not None:
            self.temperatures = [self._round_temperature(self._temp)]

    @property
    def temp_min(self) -> float:
        """Minimum temperature in temperature sweep."""
        return self._temp_min

    @temp_min.setter
    def temp_min(self, value: float):
        """Set minimum temperature (rounded if near-integer)."""
        self._temp_min = self._round_temperature(float(value))

    @property
    def temp_max(self) -> float:
        """Maximum temperature in temperature sweep."""
        return self._temp_max

    @temp_max.setter
    def temp_max(self, value: float):
        """Set maximum temperature (rounded if near-integer)."""
        self._temp_max = self._round_temperature(float(value))

    @property
    def temp_step(self) -> float:
        """Temperature step used when generating arithmetic sequence of temperatures."""
        return self._temp_step

    @temp_step.setter
    def temp_step(self, value: float):
        """Set temperature step (rounded if near-integer)."""
        self._temp_step = self._round_temperature(float(value))

    @property
    def n_temp(self) -> Optional[int]:
        """Number of temperatures for Chebyshev node generation (or None)."""
        return self._n_temp

    @n_temp.setter
    def n_temp(self, value: Optional[int]):
        """Set n_temp; convert to int if provided."""
        self._n_temp = None if value is None else int(value)

    @property
    def ascending_temp(self) -> bool:
        """Whether generated temperature list should be ascending."""
        return bool(self._ascending_temp)

    @ascending_temp.setter
    def ascending_temp(self, value: bool):
        """Set ascending_temp as boolean."""
        self._ascending_temp = bool(value)

    @property
    def n_samples_init(self) -> Optional[int]:
        """Number of samples in the first SSCHA loop (or None)."""
        return self._n_samples_init

    @n_samples_init.setter
    def n_samples_init(self, value: Optional[int]):
        """Set n_samples_init (int or None)."""
        self._n_samples_init = None if value is None else int(value)

    @property
    def n_samples_final(self) -> Optional[int]:
        """Number of samples in the second SSCHA loop (or None)."""
        return self._n_samples_final

    @n_samples_final.setter
    def n_samples_final(self, value: Optional[int]):
        """Set n_samples_final (int or None)."""
        self._n_samples_final = None if value is None else int(value)

    @property
    def tol(self) -> float:
        """Convergence tolerance used in FC calculations."""
        return self._tol

    @tol.setter
    def tol(self, value: float):
        """Set tolerance (stored as float)."""
        self._tol = float(value)

    @property
    def max_iter(self) -> int:
        """Maximum number of SSCHA iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        """Set max_iter (stored as int)."""
        self._max_iter = int(value)

    @property
    def mixing(self) -> float:
        """Mixing parameter used when updating force constants."""
        return self._mixing

    @mixing.setter
    def mixing(self, value: float):
        """Set mixing (stored as float)."""
        self._mixing = float(value)

    @property
    def mesh(self) -> tuple:
        """q-point mesh used for phonon calculations (3-tuple)."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: tuple):
        """Set mesh, validating it is a length-3 iterable."""
        if not hasattr(value, "__iter__") or len(value) != 3:
            raise ValueError("mesh must be an iterable of length 3")
        self._mesh = tuple(value)

    @property
    def init_fc_algorithm(self) -> Literal["harmonic", "file"]:
        """Algorithm used to generate initial FCs: 'harmonic' or 'file'."""
        return self._init_fc_algorithm

    @init_fc_algorithm.setter
    def init_fc_algorithm(self, value: Literal["harmonic", "file"]):
        """Set init_fc_algorithm and validate allowed values."""
        if value not in ("harmonic", "file"):
            raise ValueError("init_fc_algorithm must be 'harmonic' or 'file'")
        self._init_fc_algorithm = value

    @property
    def init_fc_file(self) -> Optional[str]:
        """Path to an HDF5 file containing initial FCs if algorithm='file'."""
        return self._init_fc_file

    @init_fc_file.setter
    def init_fc_file(self, value: Optional[str]):
        """Set init_fc_file as a string or None."""
        self._init_fc_file = None if value is None else str(value)

    @property
    def nac_params(self) -> Optional[dict]:
        """Parameters for non-analytic corrections in phonon calculations."""
        return self._nac_params

    @nac_params.setter
    def nac_params(self, value: Optional[dict]):
        """Set nac_params; convert to dict if provided."""
        self._nac_params = None if value is None else dict(value)

    @property
    def cutoff_radius(self) -> Optional[float]:
        """Optional cutoff radius for interactions."""
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, value: Optional[float]):
        """Set cutoff_radius (float or None)."""
        self._cutoff_radius = None if value is None else float(value)

    @property
    def n_unitcells(self) -> int:
        """Return number of unitcells in supercell."""
        return self._n_unitcells

    @property
    def n_atom(self) -> int:
        """Return number of atoms in supercell."""
        return self._n_atom
