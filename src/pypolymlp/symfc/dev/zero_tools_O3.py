#!/usr/bin/env python
import itertools

import numpy as np


def apply_zeros(C, zero_ids):
    """Using this function, sparse C can become larger by assigning zeros.
    Zero elements should be applied to c_trans and c_perm in constructing them.

    Warning: Slow when zero_ids is large.
    """
    """Method 1
    C[zero_ids,:] = 0
    """
    """Method 2
    C = C.tolil()
    C[zero_ids, :] = 0
    C = C.tocsr()
    """
    for i in zero_ids:
        nonzero_cols = C.getrow(i).nonzero()[1]
        for j in nonzero_cols:
            C[i, j] = 0
    return C


class FCCutoff:

    def __init__(self, supercell, cutoff=7.0):
        """Find zero FC elements inside and outside a sphere given by cutoff radius.

        Parameters
        ----------
        supercell: SymfcAtoms or PhonopyAtoms
        cutoff: Cutoff radius (in angstrom)
        """
        self.__supercell = supercell
        self.__cutoff = cutoff

    def find_zero_indices(self):
        """Find zero FC3 elements outside a sphere given by cutoff radius.

        The number of shells may be better than cutoff radius,
        because the minimal cutoff radius is strongly system-dependent.
        """
        scaled_positions = self.__supercell.scaled_positions
        n_atom = scaled_positions.shape[0]
        diff = scaled_positions[:, None, :] - scaled_positions[None, :, :]

        NN27 = n_atom**2 * 27
        N27 = n_atom * 27

        trans = np.array(list(itertools.product(*[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
        norms = np.ones((n_atom, n_atom)) * 1e10
        for t1 in trans:
            t1_tile = np.tile(t1, (n_atom, n_atom, 1))
            norms_trial = np.linalg.norm(
                (diff - t1_tile) @ self.__supercell.cell, axis=2
            )
            match = norms_trial < norms
            norms[match] = norms_trial[match]

        """Algorithm to eliminate FC3 should be reconsidered."""
        zero_atom_indices = np.array(np.where(norms > self.__cutoff)).T
        zero_atom_indices = zero_atom_indices @ np.array([NN27, N27])

        zero_indices = zero_atom_indices[:, None] + np.arange(N27)[None, :]
        zero_indices = zero_indices.reshape(-1)
        return zero_indices
