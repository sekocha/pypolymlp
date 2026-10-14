"""Class for harmonic contribution in real space."""

from typing import Optional

import numpy as np
import scipy

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .harmonic_real_base import HarmonicRealBase, const_sq_angfreq_to_sq_freq_thz
from .harmonic_utils import (
    convert_fc2_to_dynamical_matrix,
    frequencies_from_eigvals,
    schur_reduce_dynamical_matrix,
)


class HarmonicReal(HarmonicRealBase):
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
        super().__init__(
            supercell, properties, n_unitcells=n_unitcells, fc2=fc2, verbose=verbose
        )

    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        dyn = convert_fc2_to_dynamical_matrix(self._fc2, self._supercell.masses)
        square_w, eigvecs = np.linalg.eigh(dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        freq = frequencies_from_eigvals(square_w)
        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs

        self._print_frequencies(freq)
        return self._mesh_dict


class HarmonicRealReduced(HarmonicRealBase):
    """Class for harmonic contribution in real space."""

    def __init__(
        self,
        supercell: PolymlpStructure,
        properties: Properties,
        null_space_basis: np.ndarray,
        n_unitcells: Optional[int] = None,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        supercell: Supercell structure.
        properties: Properties class object to calculate energies and forces.
        null_space_basis: Basis set for null space of displacements.
        n_unitcells: Number of unitcells in supercell
        fc2: Second-order force constants.
        """
        super().__init__(
            supercell, properties, n_unitcells=n_unitcells, fc2=fc2, verbose=verbose
        )
        self._null_space_basis = null_space_basis
        reduced_basis = scipy.linalg.null_space(self._null_space_basis.T)

        N3 = null_space_basis.shape[0]
        N = N3 // 3
        null_sum = np.zeros((N3, 3))
        for i in range(N3):
            null_sum[i, i % 3] = 1.0 / np.sqrt(N)
        proj_sum = np.eye(N3) - null_sum @ null_sum.T

        proj = reduced_basis.T @ proj_sum @ reduced_basis
        eigvals, eigvecs = scipy.linalg.eigh(proj)
        eigvecs = reduced_basis @ eigvecs[:, np.isclose(eigvals, 1.0)]
        self._proj = eigvecs @ eigvecs.T

    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        dyn = convert_fc2_to_dynamical_matrix(self._fc2, self._supercell.masses)
        reduced_dyn = self._proj @ dyn @ self._proj
        square_w, eigvecs = scipy.linalg.eigh(reduced_dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        freq = frequencies_from_eigvals(square_w)
        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs

        self._print_frequencies(freq)
        return self._mesh_dict

    def _solve_eigen_equation_use_schur_complement(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix using Schur complement."""
        dyn = convert_fc2_to_dynamical_matrix(self._fc2, self._supercell.masses)
        reduced_dyn = schur_reduce_dynamical_matrix(dyn, self._null_space_basis)
        square_w, eigvecs = scipy.linalg.eigh(reduced_dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        freq = frequencies_from_eigvals(square_w)
        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs

        self._print_frequencies(freq)
        return self._mesh_dict
