"""Class for calculating electronic properties at finite temperatures."""

import numpy as np
from phonopy.qha.electron import ElectronFreeEnergy
from phonopy.units import Kb

# from phonopy.units import EvTokJmol, Kb


class ElectronProperties:
    """Class for calculating electronic properties at finite temperatures."""

    def __init__(self, eigenvalues: np.ndarray, weights: np.ndarray, n_electrons: int):
        """Init method."""
        self._efe = ElectronFreeEnergy(eigenvalues, weights, n_electrons)
        self._energies = None
        self._entropies = None
        self._free_energies = None
        self._cvs = None
        self._temperatures = None

    def run(self, temp_max: float = 1000, temp_step: float = 10):
        """Calculate properties at temperatures."""
        self._temperatures = np.arange(0.0, temp_max + 0.1, temp_step)
        self._energies = []
        self._entropies = []
        self._free_energies = []
        self._cvs = []
        for temp in self._temperatures:
            self._efe.run(T=temp)
            self._energies.append(self._efe.energy)
            self._entropies.append(self._efe.entropy)
            self._free_energies.append(self._efe.free_energy)
            self._cvs.append(self._calc_cv(temp))

        self._free_energies = np.array(self._free_energies) - self._free_energies[0]
        self._energies = np.array(self._energies) - self._energies[0]
        self._entropies = np.array(self._entropies)
        self._entropies[1:] /= self._temperatures[1:]
        self._cvs = np.array(self._cvs)
        return self

    def _calc_df_dT(self, epsilon: np.array, mu: float, T: int) -> float:
        """Calculate df/dT at temperature T."""
        de = (epsilon - mu) / (Kb * T)
        de = np.where(de < 100, de, 100.0)  # To avoid overflow
        de = np.where(de > -100, de, -100.0)  # To avoid underflow
        return (de * np.exp(de)) / (T * ((np.exp(de) + 1) ** 2))

    def _calc_cv(self, T: int) -> float:
        """Calculate Cv at temperature T.

        Return
        ------
        Cv in eV/K.
        """
        if T == 0:
            return 0.0
        self._efe.run(T)
        eigvals = self._efe._eigenvalues.reshape(len(self._efe._weights), -1)
        df_dT = self._calc_df_dT(eigvals, self._efe._mu, T)

        # TODO: Results must be tested.
        cv = self._efe._g * np.sum(
            self._efe._weights * np.sum((eigvals - self._efe._mu) * df_dT, axis=1)
        )
        return cv
        # return cv * EvTokJmol * 1000  # J/K/mol

    @property
    def free_energy(self):
        """Return free energy.

        Return
        ------
        Free energies in eV.
        """
        return self._free_energies

    @property
    def energy(self):
        """Return energy.

        Return
        ------
        Energies in eV.
        """
        return self._energies

    @property
    def entropy(self):
        """Return entropy.

        Return
        ------
        Entropies in eV/K.
        """
        return self._entropies

    @property
    def cv(self):
        """Return specific heat.

        Return
        ------
        Cv in eV/K.
        """
        return self._cvs

    @property
    def temperatures(self):
        return self._temperatures
