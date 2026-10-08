"""Class for harmonic contribution in real space."""

from typing import Optional

import numpy as np
import scipy

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .harmonic_real_base import HarmonicRealBase, const_sq_angfreq_to_sq_freq_thz


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
        fc2 = self._fc2.transpose((0, 2, 1, 3))
        size = fc2.shape[0] * fc2.shape[1]
        fc2 = np.reshape(fc2, (size, size))

        masses = np.repeat(self._supercell.masses, 3)
        masses_sqrt = np.reciprocal(np.sqrt(masses))
        dyn = (np.diag(masses_sqrt) @ fc2) @ np.diag(masses_sqrt)
        square_w, eigvecs = np.linalg.eigh(dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        negative_square_w = square_w < 0.0
        positive_square_w = square_w >= 0.0
        freq = np.zeros(square_w.shape)
        freq[positive_square_w] = np.sqrt(square_w[positive_square_w])
        freq[negative_square_w] = -np.sqrt(-square_w[negative_square_w])

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

    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        fc2 = self._fc2.transpose((0, 2, 1, 3))
        size = fc2.shape[0] * fc2.shape[1]
        fc2 = np.reshape(fc2, (size, size))

        masses = np.repeat(self._supercell.masses, 3)
        masses_sqrt = np.reciprocal(np.sqrt(masses))

        dyn = (np.diag(masses_sqrt) @ fc2) @ np.diag(masses_sqrt)
        Z = scipy.linalg.null_space(self._null_space_basis.T)
        reduced_dyn = Z.T @ dyn @ Z
        square_w, eigvecs_reduced = scipy.linalg.eigh(reduced_dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz
        eigvecs = Z @ eigvecs_reduced

        if np.any(np.abs(eigvecs.T @ self._null_space_basis) > 1e-10):
            raise RuntimeError("Eigenvectors are not in constraint null space.")

        print(dyn.shape)
        print(square_w[np.where(square_w < 0)])

        negative_square_w = square_w < 0.0
        positive_square_w = square_w >= 0.0
        freq = np.zeros(square_w.shape)
        freq[positive_square_w] = np.sqrt(square_w[positive_square_w])
        freq[negative_square_w] = -np.sqrt(-square_w[negative_square_w])

        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs
        return self._mesh_dict
