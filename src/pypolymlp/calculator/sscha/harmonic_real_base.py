"""Base Class for harmonic contribution in real space."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.core.utils import mass_table

from .harmonic_utils import (
    eliminate_outliers,
    eval_harmonic_properties,
    sample_real_space_distribution,
)

"""
Constants
---------
const_sq_angfreq_to_sq_freq_thz:
    1.602176634e-19 (eV->J) * 6.02214076e23 (avogadro) * 0.1 / (4 * pi^2)
"""
const_sq_angfreq_to_sq_freq_thz = 2.4440020137144617e2


class HarmonicRealBase(ABC):
    """Class for harmonic contribution in real space."""

    def __init__(
        self,
        supercell: PolymlpStructure,
        properties: Properties,
        n_unitcells: Optional[int] = None,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        supercell: Supercell structure.
        properties: Properties class object to calculate energies and forces.
        n_unitcells: Number of unitcells in supercell
        fc2: Second-order force constants.
        """
        self._supercell = supercell
        self._n_atom = len(supercell.elements)
        self._cartesian_positions = self._supercell.axis @ self._supercell.positions
        self.force_constants = fc2

        if n_unitcells is not None:
            self._supercell.n_unitcells = n_unitcells

        if self._supercell.n_unitcells is None:
            raise ValueError("Attribute n_unitcells required.")

        self._prop = properties
        self._verbose = verbose

        self._mesh_dict = dict()
        self._tp_dict = dict()
        self._energies_harm = None
        self._energies_full = None
        self._average_forces = None
        self._average_stress_tensor = None

        self._forces = None
        self._stress_tensors = None
        self._disps = None

        self._set_mass()
        self._e0, self._f0, self._s0 = self._prop.eval(self._supercell)
        self._ev_to_kjmol = EVtoKJmol / self._supercell.n_unitcells

    def _set_mass(self):
        """Set mass."""
        if self._supercell.masses is None:
            table = mass_table()
            masses = [table[e] for e in self._supercell.elements]
            self._supercell.masses = masses

    def _eval(self, structures: list[PolymlpStructure]):
        """Compute energies, forces, and stress tensors of structures.

        Parameters
        ----------
        structures: Structures.

        Returns
        -------
        energies: Energies measured from static energy, shape=(n_str).
        forces: Forces, shape=(n_str, 3, n_atom).
        stress_tensors: Stress tensors,
                        shape=(n_str, 6) in the order of xx, yy, zz, xy, yz, zx.
        """
        energies, forces, stress_tensors = self._prop.eval_multiple(structures)
        energies = energies - self._e0
        return np.array(energies), np.array(forces), np.array(stress_tensors)

    def run(
        self,
        temp: float = 1000,
        n_samples: int = 100,
        eliminate_outliers: bool = True,
    ):
        """Run harmonic real-space part of SSCHA.

        Parameters
        ----------
        temp: Temperature (K).
        n_samples: Number of sample structures.
        eliminate_outliers: Eliminate structures showing extreme energy values.
        """

        if self._fc2 is None:
            raise ValueError("FC2 is required for HarmonicReal.")

        self._mesh_dict = self._solve_eigen_equation()
        self._disps = self._get_distribution(temp=temp, n_samples=n_samples)
        self._supercells = get_structures_from_displacements(
            self._disps,
            self._supercell,
        )

        if self._verbose:
            print("Computing energies, forces, and stress tensors using MLP.")
        res = self._eval(self._supercells)
        self._energies_full, self._forces, self._stress_tensors = res
        self._eliminate_outliers()

        if self._verbose:
            print("Computing harmonic potentials, forces, and stress tensors.")
        self._energies_harm, hf, hs = eval_harmonic_properties(self._disps, self._fc2)
        self._compute_average_properties(hf, hs)
        return self

    @abstractmethod
    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        pass

    def _get_distribution(self, temp: float = 1000, n_samples: int = 100):
        """Calculate atomic real-space distribution from density matrix."""
        return sample_real_space_distribution(
            self._mesh_dict,
            self._supercell.masses,
            temp=temp,
            n_samples=n_samples,
        )

    def _eliminate_outliers(self, tol_negative: float = -10):
        """Eliminate outliers."""
        (
            self._disps,
            self._supercells,
            self._energies_full,
            self._forces,
            self._stress_tensors,
        ) = eliminate_outliers(
            self._disps,
            self._supercells,
            self._energies_full,
            self._forces,
            self._stress_tensors,
            tol_negative=tol_negative,
            verbose=self._verbose,
        )
        return self

    def _compute_average_properties(
        self,
        harmonic_forces: np.ndarray,
        harmonic_stress_tensors: np.ndarray,
    ):
        """Calculate average properties."""
        average_hf = np.mean(harmonic_forces, axis=0)
        average_hs = np.mean(harmonic_stress_tensors, axis=0)
        average_f = np.mean(self._forces, axis=0)
        average_s = np.mean(self._stress_tensors, axis=0)
        self._average_forces = average_f - average_hf - self._f0
        self._average_stress_tensor = average_s - average_hs - self._s0
        return self._average_forces, self._average_stress_tensor

    @property
    def force_constants(self) -> np.ndarray:
        """Return FC2, shape=(n_atom, n_atom, 3, 3)."""
        return self._fc2

    @force_constants.setter
    def force_constants(self, fc2: np.ndarray):
        """Set FC2, shape=(n_atom, n_atom, 3, 3)."""
        if fc2 is None:
            self._fc2 = None
            return
        assert fc2.shape[0] == fc2.shape[1] == self._n_atom
        assert fc2.shape[2] == fc2.shape[3] == 3
        self._fc2 = fc2

    @property
    def displacements(self) -> np.ndarray:
        """Return displacements, shape=(n_samples, 3, n_atom)."""
        if self._disps is None:
            return None
        return np.array(self._disps)

    @property
    def supercells(self) -> list[PolymlpStructure]:
        """Return supercells."""
        return self._supercells

    @property
    def forces(self) -> np.ndarray:
        """Return forces, shape=(n_samples, 3, n_atom)."""
        if self._forces is None:
            return None
        return np.array(self._forces)

    @property
    def stress_tensors(self) -> np.ndarray:
        """Return stresses, shape=(n_samples, 6)."""
        if self._stress_tensors is None:
            return None
        return np.array(self._stress_tensors)

    @property
    def full_potentials(self) -> np.ndarray:
        """Return full potentials, shape=(n_samples) in kJ/mol."""
        if self._energies_full is None:
            return None
        return self._energies_full * self._ev_to_kjmol

    @property
    def average_full_potential(self) -> float:
        """Return average full potential in kJ/mol."""
        if self._energies_full is None:
            return None
        return np.average(self._energies_full) * self._ev_to_kjmol

    @property
    def harmonic_potentials(self) -> np.ndarray:
        """Return harmonic potentials, shape=(n_samples) in kJ/mol."""
        if self._energies_harm is None:
            return None
        return self._energies_harm * self._ev_to_kjmol

    @property
    def average_harmonic_potential(self) -> float:
        """Return average harmonic potential in kJ/mol."""
        if self._energies_harm is None:
            return None
        return np.average(self._energies_harm) * self._ev_to_kjmol

    @property
    def anharmonic_potentials(self) -> np.ndarray:
        """Return anharmonic potentials, shape=(n_samples) in kJ/mol."""
        if self._energies_harm is None:
            return None
        if self._energies_full is None:
            return None
        return (self._energies_full - self._energies_harm) * self._ev_to_kjmol

    @property
    def average_anharmonic_potential(self) -> float:
        """Return average anharmonic potentials in kJ/mol."""
        if self._energies_harm is None:
            return None
        if self._energies_full is None:
            return None
        return np.average(self.anharmonic_potentials)

    @property
    def static_potential(self) -> float:
        """Return static potential of given supercell in kJ/mol."""
        return self._e0 * self._ev_to_kjmol

    @property
    def static_forces(self) -> float:
        """Return static forces of given supercell in eV/ang."""
        return self._f0

    @property
    def static_stress_tensor(self) -> float:
        """Return static stress tensor of given supercell in eV/unitcell."""
        return self._s0 / self._supercell.n_unitcells

    @property
    def average_forces(self) -> np.ndarray:
        """Return temperature-dependent forces of given supercell in eV/ang."""
        if self._average_forces is None:
            return None
        return self._average_forces

    @property
    def average_stress_tensor(self) -> np.ndarray:
        """Return temperature-dependent stress tensor in eV/unitcell."""
        if self._average_stress_tensor is None:
            return None
        return self._average_stress_tensor / self._supercell.n_unitcells

    @property
    def frequencies(self):
        """Return phonon frequencies calculated from effective harmonic H."""
        return self._mesh_dict["frequencies"]
