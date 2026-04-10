"""API Class for calculating properties from datasets on grid points."""

import os
from dataclasses import dataclass
from typing import Optional

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
        """Fit temperature-entropy data using polynomial.

        Deprecated.
        """
        if self._verbose:
            print("Temperature-Entropy fitting.", flush=True)

        self._models.st_fits = self._grid.fit_entropy_temperature(max_order=max_order)
        return self

    def eval_cp_equilibrium(self):
        """Evaluate Cp from S and Cv functions.

        Deprecated.
        """
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


@dataclass
class ThermodynamicsData:
    """Dataclass for Thermodynamics instances."""

    sscha: Thermodynamics
    sscha_el: Optional[Thermodynamics] = None
    sscha_el_ph: Optional[Thermodynamics] = None
    ti: Optional[Thermodynamics] = None
    ti_el: Optional[Thermodynamics] = None
    ti_el_ph: Optional[Thermodynamics] = None
    ti_ext: Optional[Thermodynamics] = None
    ti_ext_el: Optional[Thermodynamics] = None
    ti_ext_el_ph: Optional[Thermodynamics] = None

    def _run_standard(self, th: Thermodynamics):
        """Use a standard fitting procedure."""
        th.fit_free_energy_volume()
        th.fit_entropy_volume(max_order=6)
        th.eval_entropy_equilibrium()
        th.eval_cp_numerical()
        return th

    def _run_deprecated(self, th: Thermodynamics):
        """Use a standard but deprecated fitting procedure."""
        th.fit_free_energy_volume()
        th.fit_entropy_volume(max_order=6)
        th.eval_entropy_equilibrium()
        th.fit_entropy_temperature(max_order=4)
        th.fit_cv_volume(max_order=4)
        th.eval_cp_equilibrium()
        return th

    def run(self, verbose: bool = False):
        """Run thermodynamic property estimation."""
        if verbose:
            print("# ------- SSCHA ------- #", flush=True)
        self.sscha = self._run_standard(self.sscha)

        if self.sscha_el is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.sscha_el = self._run_standard(self.sscha_el)

        if self.sscha_el_ph is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.sscha_el_ph = self._run_standard(self.sscha_el_ph)

        if self.ti is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti = self._run_standard(self.ti)

        if self.ti_el is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti_el = self._run_standard(self.ti_el)

        if self.ti_el_ph is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti_el_ph = self._run_standard(self.ti_el_ph)

        if self.ti_ext is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti_ext = self._run_standard(self.ti_ext)

        if self.ti_ext_el is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti_ext_el = self._run_standard(self.ti_ext_el)

        if self.ti_ext_el_ph is not None:
            if verbose:
                print("# ------- SSCHA ------- #", flush=True)
            self.ti_ext_el_ph = self._run_standard(self.ti_ext_el_ph)
        return self

    def save(self, path: str = "polymlp_thermodynamics"):
        """Save properties."""
        os.makedirs(path, exist_ok=True)
        name = path + "/sscha.yaml"
        self.sscha.save_thermodynamics_yaml(filename=name)

        if self.sscha_el is not None:
            name = path + "/sscha_el.yaml"
            self.sscha_el.save_thermodynamics_yaml(filename=name)

        if self.sscha_el_ph is not None:
            name = path + "/sscha_el_ph.yaml"
            self.sscha_el_ph.save_thermodynamics_yaml(filename=name)

        if self.ti is not None:
            name = path + "/ti.yaml"
            self.ti.save_thermodynamics_yaml(filename=name)

        if self.ti_el is not None:
            name = path + "/ti_el.yaml"
            self.ti_el.save_thermodynamics_yaml(filename=name)

        if self.ti_el_ph is not None:
            name = path + "/ti_el_ph.yaml"
            self.ti_el_ph.save_thermodynamics_yaml(filename=name)

        if self.ti_ext is not None:
            name = path + "/ti_ext.yaml"
            self.ti_ext.save_thermodynamics_yaml(filename=name)

        if self.ti_ext_el is not None:
            name = path + "/ti_ext_el.yaml"
            self.ti_ext_el.save_thermodynamics_yaml(filename=name)

        if self.ti_ext_el_ph is not None:
            name = path + "/ti_ext_el_ph.yaml"
            self.ti_ext_el_ph.save_thermodynamics_yaml(filename=name)
