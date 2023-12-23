#!/usr/bin/env python
import numpy as np
from scipy.sparse import coo_array, kron

from phonopy.structure.atoms import PhonopyAtoms
from symfc.spg_reps import SpgReps

class SpgRepsO4(SpgReps):
    """Class of reps of space group operations for fc4."""

    def __init__(self, supercell: PhonopyAtoms):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.

        """
        self._r4_reps: list[coo_array]
        self._col: np.ndarray
        self._data: np.ndarray
        super().__init__(supercell)

    @property
    def r_reps(self) -> list[coo_array]:
        """Return 4th rank tensor rotation matricies."""
        return self._r4_reps

    def get_sigma4_rep(self, i: int) -> coo_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma4_rep_data(i)
        return coo_array((data, (row, col)), shape=shape)

    def _prepare(self):
        rotations = super()._prepare()
        N = len(self._numbers)
        a = np.arange(N)
        self._atom_quadruplets = np.stack(np.meshgrid(a,a,a,a), 
                                          axis=-1).reshape(-1, 4)
        self._coeff = np.array([1, N, N**2, N**3], dtype=int)
        self._col = self._atom_quadruplets @ self._coeff
        self._data = np.ones(N * N * N * N, dtype=int)
        self._compute_r4_reps(rotations)

    def _compute_r4_reps(self, rotations: np.ndarray, tol: float = 1e-10):
        """Compute and return 4th rank tensor rotation matricies."""
        uri = self._unique_rotation_indices
        r4_reps = []
        for r in rotations[uri]:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r4_rep = np.kron(r_c, np.kron(r_c, np.kron(r_c, r_c)))
            row, col = np.nonzero(np.abs(r4_rep) > tol)
            data = r4_rep[(row, col)]
            r4_reps.append(coo_array((data, (row, col)), shape=r4_rep.shape))
        self._r4_reps = r4_reps

    def _get_sigma4_rep_data(self, i: int) -> coo_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NNNN = len(self._numbers) ** 4
        row = permutation[self._atom_quadruplets] @ self._coeff
        return self._data, row, self._col, (NNNN, NNNN)

