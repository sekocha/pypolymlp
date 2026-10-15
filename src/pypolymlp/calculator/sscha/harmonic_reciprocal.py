"""Class for harmonic contribution in reciprocal space."""

from typing import Optional

import numpy as np
import scipy
from phonopy import Phonopy

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_utils import frequencies_from_eigvals
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

from .harmonic_reciprocal_base import (
    HarmonicReciprocalBase,
    const_sq_angfreq_to_sq_freq_thz,
)


class HarmonicReciprocal(HarmonicReciprocalBase):
    """Class for harmonic contribution in reciprocal space."""

    def __init__(
        self,
        phonopy_obj: Phonopy,
        properties: Properties,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        phonopy_obj: Phonopy class object.
        property: Property class object to calculate energies and forces.
        fc2: Second-order force constants.
        """
        super().__init__(phonopy_obj, properties, fc2=fc2, verbose=verbose)

    def run(
        self,
        qmesh: tuple = (10, 10, 10),
        temp: float = 1000,
        only_frequencies: bool = False,
        hide_imaginary: bool = False,
    ):
        """Calculate frequencies and properties."""
        self._ph.run_mesh(qmesh)
        self._frequencies = self._ph._mesh.frequencies
        self._mesh_dict = self._ph.get_mesh_dict()
        self._print_frequencies()
        self._ph.run_total_dos()

        if only_frequencies:
            return self._frequencies

        free_energy_raw, free_energy_rev = self._compute_thermal_properties(
            temp=temp, hide_imaginary=hide_imaginary
        )
        return free_energy_raw, free_energy_rev


class HarmonicReciprocalReduced(HarmonicReciprocalBase):
    """Class for harmonic contribution in reciprocal space."""

    def __init__(
        self,
        phonopy_obj: Phonopy,
        properties: Properties,
        null_space_basis: np.ndarray,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        phonopy_obj: Phonopy class object.
        property: Property class object to calculate energies and forces.
        null_space_basis: Basis set for null space of displacements.
        fc2: Second-order force constants.
        """
        super().__init__(phonopy_obj, properties, fc2=fc2, verbose=verbose)
        self._unitcell = phonopy_cell_to_structure(phonopy_obj.unitcell)
        self._null_space_basis = null_space_basis

        if self._null_space_basis.shape[0] != len(self._unitcell.elements) * 3:
            raise RuntimeError("Size of null space basis not appropriate.")

        # TODO: Need to check sum = 0
        self._reduced_basis = scipy.linalg.null_space(self._null_space_basis.T)
        self._proj_reduced = self._reduced_basis @ self._reduced_basis.T

        # N = len(self._unitcell.elements)
        # N3 = 3 * N
        # null_sum = np.zeros((N3, 3))
        # for i in range(N3):
        #     null_sum[i, i % 3] = 1.0 / np.sqrt(N)
        # self._proj_sum = np.eye(N3) - null_sum @ null_sum.T

        # proj = self._reduced_basis.T @ self._proj_sum @ self._reduced_basis
        # eigvals, eigvecs = scipy.linalg.eigh(proj)
        # eigvecs = self._reduced_basis @ eigvecs[:, np.isclose(eigvals, 1.0)]
        # self._proj_reduced = eigvecs @ eigvecs.T

    def _solve_eigen_equation(self, qpoints: np.ndarray) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        # TODO: Check units in dynamical matrix used in phonopy.
        square_freqs = []
        for q in qpoints:
            prod = self._unitcell.positions.T @ q
            phase = np.repeat(np.exp(2 * np.pi * 1j * prod), 3)
            proj1 = phase[:, None] * self._proj_reduced * phase.conj()[None, :]

            # # proj_sum = phase[:, None] * self._proj_sum * phase.conj()[None, :]
            # # proj1 = proj1 @ proj_sum @ proj1
            # proj1 = proj1 @ self._proj_sum @ proj1

            eigvals, eigvecs = scipy.linalg.eigh(proj1)
            eigvecs = eigvecs[:, np.isclose(eigvals, 1.0)]
            proj_red = eigvecs @ eigvecs.conj().T

            dyn = self._ph.get_dynamical_matrix_at_q(q)
            # reduced_dyn = proj_red @ dyn @ proj_red.conj().T
            reduced_dyn = proj_red.conj().T @ dyn @ proj_red
            square_w, eigvecs = scipy.linalg.eigh(reduced_dyn)
            square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz
            square_freqs.append(square_w)

        freqs = frequencies_from_eigvals(np.array(square_freqs))
        # freqs[(freqs < 0.1) & (freqs > -0.01)] = -0.1
        return freqs

    def run(
        self,
        qmesh: tuple = (10, 10, 10),
        temp: float = 1000,
        only_frequencies: bool = False,
        hide_imaginary: bool = False,
    ):
        """Calculate frequencies and properties."""
        self._ph.run_mesh(qmesh)
        self._mesh_dict = self._ph.get_mesh_dict()
        self._frequencies = self._solve_eigen_equation(self._mesh_dict["qpoints"])
        self._print_frequencies()

        # TODO: setter is correct?
        # self._ph._mesh._frequencies = self._frequencies
        self._ph.run_total_dos()

        if only_frequencies:
            return self._frequencies

        free_energy_raw, free_energy_rev = self._compute_thermal_properties(
            temp=temp, hide_imaginary=hide_imaginary
        )
        return free_energy_raw, free_energy_rev
