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
        if np.any(~np.isclose(np.sum(self._null_space_basis, axis=0), 0.0)):
            raise RuntimeError("Sum of each null space basis must be zero.")

        self._reduced_basis = scipy.linalg.null_space(self._null_space_basis.T)
        self._proj_reduced = self._reduced_basis @ self._reduced_basis.T

    def _projector_dynamical_matrix(self, q: np.ndarray):
        """Calculate complex projector for dynamical matrix."""
        prod = self._unitcell.positions.T @ q
        phase = np.repeat(np.exp(2 * np.pi * 1j * prod), 3)
        proj1 = phase.conj()[:, None] * self._proj_reduced * phase[None, :]
        return proj1, phase

    def _solve_eigen_equation(self, qpoints: np.ndarray) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        # TODO: Check units in dynamical matrix used in phonopy.
        square_freqs = []
        for q in qpoints:
            proj_red, phase = self._projector_dynamical_matrix(q)
            dyn = self._ph.get_dynamical_matrix_at_q(q)
            reduced_dyn = proj_red @ dyn @ proj_red
            square_w, eigvecs = scipy.linalg.eigh(reduced_dyn)
            square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

            inner_prod = eigvecs.T @ (phase[:, None] * self._null_space_basis)
            ortho_rows = np.where(np.all(np.isclose(inner_prod, 0.0), axis=1))[0]
            square_freqs.append(square_w[ortho_rows])

        freqs = frequencies_from_eigvals(np.array(square_freqs))
        return freqs

    def run(
        self,
        qmesh: tuple = (10, 10, 10),
        temp: float = 1000,
        only_frequencies: bool = False,
        hide_imaginary: bool = False,
    ):
        """Calculate frequencies and properties."""
        self._ph.init_mesh(qmesh)
        self._frequencies = self._solve_eigen_equation(self._ph.mesh.qpoints)

        # TODO: Setter is correct?,
        # TODO: Reduced number of frequencies can be assigned to mesh._frequencies?
        self._ph._mesh._frequencies = self._frequencies
        self._mesh_dict = self._ph.get_mesh_dict()
        self._print_frequencies()
        self._ph.run_total_dos()

        if only_frequencies:
            return self._frequencies

        free_energy_raw, free_energy_rev = self._compute_thermal_properties(
            temp=temp, hide_imaginary=hide_imaginary
        )
        return free_energy_raw, free_energy_rev
