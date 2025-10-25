"""Utility functions for EOS fitting."""

import numpy as np
from phonopy.qha.core import BulkModulus
from phonopy.units import EVAngstromToGPa


class EOS:
    """class for EOS fitting."""

    def __init__(self, volumes: np.ndarray, energies: np.ndarray):
        """Init method."""
        self._bm = BulkModulus(volumes=volumes, energies=energies, eos="vinet")

    def eval(self, volumes: np.array):
        """Evaluate energy values for given volumes."""
        return self._bm._eos(volumes, *self._bm.get_parameters())

    def eval_gibbs_pressure(self, volumes: np.ndarray, eps: float = 1e-4):
        """Transform Helmholtz free energy to Gibbs free energy.

        Return
        ------
        gibbs_free_energies: Array of (pressure in GPa, Gibbs free energy).
        """
        gibbs_free_energies = []
        free_energies = self.eval(volumes)
        for vol, fe in zip(volumes, free_energies):
            eos_f, eos_b = self.eval([vol + eps, vol - eps])
            deriv = (eos_f - eos_b) / (2 * eps)
            press = -deriv  # in eV/ang^3
            gibbs = fe + press * vol
            press_gpa = press * EVAngstromToGPa  # in GPa
            gibbs_free_energies.append([press_gpa, gibbs])
        return gibbs_free_energies

    @property
    def b0(self):
        """Return bulk modulus in GPa."""
        return self._bm.bulk_modulus * EVAngstromToGPa

    @property
    def e0(self):
        """Return equilibrium energy."""
        return self._bm.energy

    @property
    def v0(self):
        """Return equilibrium volume."""
        return self._bm.equilibrium_volume
