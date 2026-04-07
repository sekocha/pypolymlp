"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from phonopy.physical_units import get_physical_units

from pypolymlp.core.units import EVtoJmol

units = get_physical_units()
EVAngstromToGPa = units.EVAngstromToGPa


@dataclass
class FittedModels:
    """Dataclass for fitted thermodynamics functions."""

    volumes: np.ndarray
    temperatures: np.ndarray
    fv_fits: Optional[list] = None
    sv_fits: Optional[list] = None
    cv_fits: Optional[list] = None
    ft_fits: Optional[list] = None
    st_fits: Optional[list] = None
    et_fits: Optional[list] = None

    def __post_init__(self):
        """Post init method."""
        self.volume_threshold = max(self.volumes) * 1.2
        self.bm_threshold = 1e-3

    def reshape(self, ix_v: np.ndarray, ix_t: np.ndarray):
        """Reshape objects with common grid."""
        self.volumes = self.volumes[ix_v]
        self.temperatures = self.temperatures[ix_t]
        if self.fv_fits is not None:
            self.fv_fits = [self.fv_fits[i] for i in ix_t]
        if self.sv_fits is not None:
            self.sv_fits = [self.sv_fits[i] for i in ix_t]
        if self.cv_fits is not None:
            self.cv_fits = [self.cv_fits[i] for i in ix_t]
        if self.ft_fits is not None:
            self.ft_fits = [self.ft_fits[i] for i in ix_v]
        if self.st_fits is not None:
            self.st_fits = [self.st_fits[i] for i in ix_v]
        if self.et_fits is not None:
            self.et_fits = [self.et_fits[i] for i in ix_v]
        return self

    def extract(self, itemp: int):
        """Retrun fitted functions for at a temperature index."""
        fv = self.fv_fits[itemp] if self.fv_fits is not None else None
        sv = self.sv_fits[itemp] if self.sv_fits is not None else None
        cv = self.cv_fits[itemp] if self.cv_fits is not None else None
        return fv, sv, cv

    def eval_eq_free_energy(self, itemp: int):
        """Evaluate entropy at equilibrium volume."""
        if self.fv_fits is None:
            raise RuntimeError("F-V functions not found.")
        if self.fv_fits[itemp] is None:
            return None

        if self.fv_fits[itemp].v0 > self.volume_threshold:
            return None
        if self.fv_fits[itemp].b0 / EVAngstromToGPa < self.bm_threshold:
            return None

        return self.fv_fits[itemp].eval(self.fv_fits[itemp].v0)

    def eval_eq_entropy(self, itemp: int):
        """Evaluate entropy at equilibrium volume."""
        if self.fv_fits is None:
            raise RuntimeError("F-V functions not found.")
        if self.sv_fits is None:
            raise RuntimeError("S-V functions not found.")
        if self.fv_fits[itemp] is None or self.sv_fits[itemp] is None:
            return None

        if self.fv_fits[itemp].v0 > self.volume_threshold:
            return None
        if self.fv_fits[itemp].b0 / EVAngstromToGPa < self.bm_threshold:
            return None

        return self.sv_fits[itemp].eval(self.fv_fits[itemp].v0)

    def eval_eq_cv(self, itemp: int):
        """Evaluate Cv contribution at equilibrium volume."""
        if self.fv_fits is None:
            raise RuntimeError("F-V functions not found.")
        if self.cv_fits is None:
            raise RuntimeError("Cv-V functions not found.")
        if self.fv_fits[itemp] is None or self.cv_fits[itemp] is None:
            return None

        if self.fv_fits[itemp].v0 > self.volume_threshold:
            return None
        if self.fv_fits[itemp].b0 / EVAngstromToGPa < self.bm_threshold:
            return None

        return self.cv_fits[itemp].eval(self.fv_fits[itemp].v0)

    def eval_eq_cp(self, itemp: int):
        """Evaluate Cp at equilibrium volume."""
        cv_val = self.eval_eq_cv(itemp)
        if cv_val is None:
            return None

        try:
            fv, sv, _ = self.extract(itemp)
            v0, b0 = fv.v0, fv.b0
            s_deriv = sv.eval_derivative(v0)
            temp = self.temperatures[itemp]
            bm = b0 / EVAngstromToGPa

            if bm < self.bm_threshold:
                return None
            if v0 > self.volume_threshold:
                return None

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
        if self.fv_fits is None:
            raise RuntimeError("F-V functions not found.")
        return np.array([fv.eval(volumes) for fv in self.fv_fits]).T

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
        if self.fv_fits is None:
            raise RuntimeError("F-V functions not found.")
        return np.array([fv.eval_gibbs_pressure(volumes) for fv in self.fv_fits])


def sum_matrix_data(matrix1: np.ndarray, matrix2: np.ndarray):
    """Calculate sum of two matrices."""
    if matrix1.shape != matrix2.shape:
        raise RuntimeError("Inconsistent matrix shape.")
    res = np.full(matrix1.shape, None)
    mask = np.equal(matrix1, None) | np.equal(matrix2, None)
    res[~mask] = matrix1[~mask] + matrix2[~mask]
    return res
