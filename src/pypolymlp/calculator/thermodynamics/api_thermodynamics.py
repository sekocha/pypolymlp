"""API Class for calculating properties from datasets on grid points."""

import numpy as np

from pypolymlp.calculator.thermodynamics.thermodynamics_io import (
    save_thermodynamics_yaml,
)
from pypolymlp.calculator.thermodynamics.thermodynamics_parser import GridVT
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import FittedModels
from pypolymlp.core.units import EVtoJmol


class Thermodynamics:
    """API Class for calculating properties from datasets on grid points."""

    def __init__(self, grid: GridVT, verbose: bool = False):
        """Init method.

        Input data units
        ----------------
        free_energy: eV/atom
        entropy: eV/K/atom
        heat_capacity: J/K/mol (/Avogadro's number of atoms)
        """
        self._grid = grid
        self._grid._verbose = verbose
        self._verbose = verbose

        self._volumes = self._grid.volumes
        self._temperatures = self._grid.temperatures
        self._models = FittedModels(self._volumes, self._temperatures)

        self._eq_free_energies = None
        self._eq_entropies = None
        self._eq_cp = None

    def fit_free_energy_volume(self):
        """Fit volume-free energy data to Vinet EOS."""
        if self._verbose:
            print("Volume-FreeEnergy fitting.", flush=True)

        self._models.fv_fits = self._grid.fit_free_energy_volume()
        return self

    def fit_entropy_volume(self, max_order: int = 6):
        """Fit volume-entropy data using polynomial."""
        if self._verbose:
            print("Volume-Entropy fitting.", flush=True)

        self._models.sv_fits = self._grid.fit_entropy_volume(max_order=max_order)
        return self

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        if self._verbose:
            print("Volume-Cv fitting.", flush=True)

        self._models.cv_fits = self._grid.fit_cv_volume(max_order=max_order)
        return self

    def eval_entropy_equilibrium(self):
        """Evaluate entropies at equilibrium volumes."""
        self._eq_entropies = np.array(
            [self._models.eval_eq_entropy(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_entropies

    def eval_cp_numerical(self):
        """Evaluate Cp from S_eq numerically."""
        if self._eq_entropies is None:
            raise RuntimeError("Entropy at equilibrium volumes not found.")
        n_temps = len(self._temperatures)
        self._eq_cp = np.zeros(n_temps)
        for i in range(1, len(self._temperatures) - 1):
            dS = self._eq_entropies[i + 1] - self._eq_entropies[i - 1]
            dT = self._temperatures[i + 1] - self._temperatures[i - 1]
            cp = (dS / dT) * self._temperatures[i] * EVtoJmol
            self._eq_cp[i] = cp

        self._eq_cp[-1] = None
        if not np.isclose(self._temperatures[0], 0.0):
            self._eq_cp[0] = None
        return self._eq_cp

    def fit_entropy_temperature(self, max_order: int = 4):
        """Fit temperature-entropy data using polynomial."""
        if self._verbose:
            print("Temperature-Entropy fitting.", flush=True)

        self._models.st_fits = self._grid.fit_entropy_temperature(max_order=max_order)
        return self

    def eval_cp_equilibrium(self):
        """Evaluate Cp from S and Cv functions."""
        self._eq_cp = np.array(
            [self._models.eval_eq_cp(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_cp

    def save_thermodynamics_yaml(self, filename: str = "polymlp_thermodynamics.yaml"):
        """Save fitted thermodynamics properties."""
        save_thermodynamics_yaml(
            self._volumes,
            self._temperatures,
            self._models,
            self._eq_entropies,
            self._eq_cp,
            self._grid.get_properties("free_energy"),
            filename=filename,
        )
        return self

    def eval_free_energies(self, volumes: np.ndarray):
        """Return free energy.

        Return
        ------
        Helmholtz free energies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._models.eval_helmholtz_free_energies(volumes)

    def eval_entropies(self, volumes: np.ndarray):
        """Return entropies.

        Return
        ------
        Entropies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._models.eval_entropies(volumes)

    def eval_heat_capacities(self, volumes: np.ndarray):
        """Return heat capacities.

        Return
        ------
        Heat capacities.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        if not self._is_heat_capacity:
            return None
        return self._models.eval_heat_capacities(volumes)

    def eval_gibbs_free_energies(self, volumes: np.ndarray):
        """Return Gibbs free energy.

        Return
        ------
        Gibbs free energies.
            Array of (temperature index, pressure in GPa, Gibbs free energy).
        """
        return self._models.eval_gibbs_free_energies(volumes)

    @property
    def volumes(self):
        """Return volumes."""
        return self._grid.volumes

    @property
    def temperatures(self):
        """Return temperatures."""
        return self._grid.temperatures

    @property
    def data(self):
        """Return data."""
        return self._grid.data

    @property
    def grid(self):
        """Return grid points in GridVT.

        Rows and columns of data correspond to volumes and temperatures, respectively.
        """
        return self._grid

    @property
    def fitted_models(self):
        """Return fitted models."""
        return self._models


#     def calculate_reference(self):
#         """Calculate reference properties."""
#         if self._verbose:
#             print("Calculate reference properties.", flush=True)
#
#         for g in self._grid:
#             g = calculate_reference(g, self._temperatures)
#         return self
#
#     def copy_reference(self, grid: np.ndarray):
#         """Copy reference properties."""
#         for g1, g2 in zip(grid, self._grid):
#             for p, q in zip(g1, g2):
#                 if q is not None:
#                     q.copy_reference(p)
#         return self
#
#     def calculate_harmonic_free_energies(self):
#         """Calculate harmonic free energies."""
#         if self._verbose:
#             print("Calculate harmonic free energies.", flush=True)
#
#         for g, vol in zip(self._grid, self._volumes):
#             if self._verbose:
#                 print("- Volume:", np.round(vol, 3), flush=True)
#             g = calculate_harmonic_free_energies(g)
#         return self
