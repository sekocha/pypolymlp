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

    def reset_reference(self):
        """Reset data."""
        self.reference_free_energy = None
        self.reference_entropy = None
        self.reference_heat_capacity = None
        return self

    def add(self, gp_data):
        """Add data."""
        if self.free_energy is not None and gp_data.free_energy is not None:
            self.free_energy += gp_data.free_energy
        else:
            self.free_energy = None
        if self.entropy is not None and gp_data.entropy is not None:
            self.entropy += gp_data.entropy
        else:
            self.entropy = None
        if self.heat_capacity is not None and gp_data.heat_capacity is not None:
            self.heat_capacity += gp_data.heat_capacity
        else:
            self.heat_capacity = None
        if self.restart is None:
            self.restart = gp_data.restart
        if self.path_fc2 is None:
            self.path_fc2 = gp_data.path_fc2
        self.reset_reference()
        return self


@dataclass
class FittedModels:
    """Dataclass for fitted thermodynamics functions."""

    volumes: np.ndarray
    temperatures: np.ndarray
    eos_fits: Optional[list] = None
    sv_fits: Optional[list] = None
    cv_fits: Optional[list] = None
    ft_fits: Optional[list] = None
    st_fits: Optional[list] = None

    def reshape(self, ix_v: np.ndarray, ix_t: np.ndarray):
        """Reshape objects with common grid."""
        self.volumes = self.volumes[ix_v]
        self.temperatures = self.temperatures[ix_t]
        if self.eos_fits is not None:
            self.eos_fits = [self.eos_fits[i] for i in ix_t]
        if self.sv_fits is not None:
            self.sv_fits = [self.sv_fits[i] for i in ix_t]
        if self.cv_fits is not None:
            self.cv_fits = [self.cv_fits[i] for i in ix_t]
        if self.ft_fits is not None:
            self.ft_fits = [self.ft_fits[i] for i in ix_v]
        if self.st_fits is not None:
            self.st_fits = [self.st_fits[i] for i in ix_v]
        return self

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
        try:
            eos, sv, _ = self.extract(itemp)
            v0, b0 = eos.v0, eos.b0
            s_deriv = sv.eval_derivative(v0)
            temp = self.temperatures[itemp]
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


def compare_conditions(array1: np.ndarray, array2: np.ndarray):
    """Return indices with the same values in two arrays"""
    ids1, ids2 = [], []
    for i1, val in enumerate(array1):
        i2 = np.where(np.isclose(array2, val))[0]
        if len(i2) > 0:
            ids1.append(i1)
            ids2.append(i2[0])
    return np.array(ids1), np.array(ids2)


def get_common_grid(
    volumes1: np.ndarray,
    volumes2: np.ndarray,
    temperatures1: np.ndarray,
    temperatures2: np.ndarray,
):
    """Return common grid for two conditions."""
    ids1_v, ids2_v = compare_conditions(volumes1, volumes2)
    ids1_t, ids2_t = compare_conditions(temperatures1, temperatures2)
    return (ids1_v, ids1_t), (ids2_v, ids2_t)


def sum_matrix_data(matrix1: np.ndarray, matrix2: np.ndarray):
    """Calculate sum of two matrices."""
    if matrix1.shape != matrix2.shape:
        raise RuntimeError("Inconsistent matrix shape.")
    res = np.full(matrix1.shape, None)
    mask = np.equal(matrix1, None) | np.equal(matrix2, None)
    res[~mask] = matrix1[~mask] + matrix2[~mask]
    return res
