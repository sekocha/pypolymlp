"""Class for harmonic contribution in real space."""

from typing import Optional

import numpy as np
import scipy

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .harmonic_real_base import HarmonicRealBase, const_sq_angfreq_to_sq_freq_thz
from .harmonic_utils import (  # convert_dynamical_matrix_to_fc2,
    convert_fc2_to_dynamical_matrix,
    frequencies_from_eigvals,
    reduce_dynamical_matrix,
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

        if self._verbose:
            tol = 0.001
            print("Imaginary frequencies:")
            print(square_w[square_w < -tol])
            print("Zero frequencies:")
            print(square_w[np.abs(square_w) < tol])

        freq = frequencies_from_eigvals(square_w)
        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs
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
        self._Z = scipy.linalg.null_space(self._null_space_basis.T)

    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        dyn = convert_fc2_to_dynamical_matrix(self._fc2, self._supercell.masses)
        reduced_dyn = reduce_dynamical_matrix(dyn, self._null_space_basis)
        # self._fc2 = convert_dynamical_matrix_to_fc2(reduced_dyn, self._supercell.masses)
        square_w, eigvecs = scipy.linalg.eigh(reduced_dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        if self._verbose:
            tol = 0.001
            print("Imaginary frequencies:")
            print(square_w[square_w < -tol])
            print("Zero frequencies:")
            print(square_w[np.abs(square_w) < tol])

        freq = frequencies_from_eigvals(square_w)
        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs
        return self._mesh_dict


#     def _solve_eigen_equation(self) -> dict:
#         """Solve eigenvalue equation for dynamical matrix."""
#         dyn = convert_fc2_to_dynamical_matrix(self._fc2, self._supercell.masses)
#         reduced_dyn = self._Z.T @ dyn @ self._Z
#         square_w, eigvecs_reduced = scipy.linalg.eigh(reduced_dyn)
#         square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz
#         eigvecs = self._Z @ eigvecs_reduced
#
#         if np.any(np.abs(eigvecs.T @ self._null_space_basis) > 1e-10):
#             raise RuntimeError("Eigenvectors are not in constraint null space.")
#
#         if self._verbose:
#            tol = 0.001
#            print("Imaginary frequencies:")
#            print(square_w[square_w < -tol])
#            print("Zero frequencies:")
#            print(square_w[np.abs(square_w) < tol])
#
#         freq = frequencies_from_eigvals(square_w)
#         self._mesh_dict["frequencies"] = freq
#         self._mesh_dict["eigenvectors"] = eigvecs
#         return self._mesh_dict
