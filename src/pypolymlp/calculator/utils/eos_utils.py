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
