"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.core.units import EVtoJmol


@dataclass
class GridPointData:
    """Dataclass for properties on a volume-temperature grid point."""

    volume: float
    temperature: float
    data_type: Optional[Literal["sscha", "ti", "electron"]] = None
    restart: Optional[Restart] = None
    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None

    reference_free_energy: Optional[float] = None
    reference_entropy: Optional[float] = None
    reference_heat_capacity: Optional[float] = None

    path_yaml: Optional[float] = None
    path_fc2: Optional[float] = None

    def reset(self):
        """Reset data."""
        self.harmonic_heat_capacity = None
        self.reference_free_energy = None
        self.reference_entropy = None
        self.reference_heat_capacity = None
        self.path_yaml = None
        self.path_fc2 = None
        return self


@dataclass
class FittedModels:
    """Dataclass for fitted thermodynamics functions."""

    volumes: np.ndarray
    temperatures: np.ndarray
    eos_fits: Optional[list] = None
    sv_fits: Optional[list] = None
    st_fits: Optional[list] = None
    cv_fits: Optional[list] = None

    def extract(self, itemp: int):
        """Retrun fitted functions for at a temperature index."""
        eos = self.eos_fits[itemp] if self.eos_fits is not None else None
        sv = self.sv_fits[itemp] if self.sv_fits is not None else None
        cv = self.cv_fits[itemp] if self.cv_fits is not None else None
        return eos, sv, cv

    def eval_eq_entropy(self, itemp: int):
        """Evaluate entropy at equilibrium volume."""
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        if self.sv_fits is None:
            raise RuntimeError("S-V functions not found.")
        if self.eos_fits[itemp] is None or self.sv_fits[itemp] is None:
            return None

        return self.sv_fits[itemp].eval(self.eos_fits[itemp].v0)

    def eval_eq_cv(self, itemp: int):
        """Evaluate Cv contribution at equilibrium volume."""
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        if self.cv_fits is None:
            raise RuntimeError("Cv-V functions not found.")
        if self.eos_fits[itemp] is None or self.cv_fits[itemp] is None:
            return None
        return self.cv_fits[itemp].eval(self.eos_fits[itemp].v0)

    def eval_eq_cp(self, itemp: int):
        """Evaluate Cp at equilibrium volume."""
        cv_val = self.eval_eq_cv(itemp)
        eos, sv, _ = self.extract(itemp)
        v0, b0 = eos.v0, eos.b0
        s_deriv = sv.eval_derivative(v0)
        temp = self.temperatures[itemp]
        try:
            bm = b0 / EVAngstromToGPa
            add = temp * v0 * (s_deriv**2) / bm
            add *= EVtoJmol
        except:
            return None

        return cv_val + add

    def eval_helmholtz_free_energies(self, volumes: np.ndarray):
        """Return free energy.

        Return
        ------
        Helmholtz free energies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        return np.array([eos.eval(volumes) for eos in self.eos_fits]).T

    def eval_entropies(self, volumes: np.ndarray):
        """Return entropies.

        Return
        ------
        Entropies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        if self.sv_fits is None:
            raise RuntimeError("S-V functions not found.")
        return np.array([sv.eval(volumes) for sv in self.sv_fits]).T

    def eval_heat_capacities(self, volumes: np.ndarray):
        """Return heat capacities.

        Return
        ------
        Heat capacities.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        if self.cv_fits is None:
            raise RuntimeError("Cv-V functions not found.")
        return np.array([cv.eval(volumes) for cv in self.cv_fits]).T

    def eval_gibbs_free_energies(self, volumes: np.ndarray):
        """Return Gibbs free energy.

        Return
        ------
        Gibbs free energies.
            Array of (temperature index, pressure in GPa, Gibbs free energy).
        """
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        return np.array([eos.eval_gibbs_pressure(volumes) for eos in self.eos_fits])
