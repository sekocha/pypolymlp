"""Utility functions for 3rd order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron
from symfc.spg_reps import SpgRepsO3
from symfc.utils.cutoff_tools_O3 import FCCutoffO3
from symfc.utils.utils import get_indep_atoms_by_lat_trans


def get_atomic_lat_trans_decompr_indices_sparse_O3(
    trans_perms: np.ndarray, fc_cutoff: FCCutoffO3
) -> np.ndarray:
    """Return indices to de-compress compressed matrix by atom-lat-trans-sym.

    This is atomic permutation only version of get_lat_trans_decompr_indices.

    Usage
    -----
    vec[indices] of shape (n_a*N*N,) gives an array of shape=(N**3,).
    1/sqrt(n_lp) must be multiplied manually after decompression.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.

    Returns
    -------
    indices : ndarray
        Indices of n_a * N * N elements.
        shape=(N**3,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N**3

    n = 0
    indices = np.ones(size_row, dtype="int_") * -1
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**2
        for j in range(N):
            index_shift_j = index_shift_i + trans_perms[:, j] * N
            for k in range(N):
                index_shift = index_shift_j + trans_perms[:, k]
                indices[index_shift] = n
                n += 1

    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**2
        for j in fc_cutoff.outsides[i_patom]:
            index_shift_j = index_shift_i + trans_perms[:, j] * N
            for k in fc_cutoff.outsides[i_patom]:
                index_shift = index_shift_j + trans_perms[:, k]
                indices[index_shift] = -1

    return indices


def get_compr_coset_reps_sum_O3_sparse(
    spg_reps: SpgRepsO3,
    fc_cutoff: FCCutoffO3,
    atomic_decompr_idx=None,
    c_pt: csr_array = None,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**3 * 27 // n_lp if c_pt is None else c_pt.shape[1]
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")

    if atomic_decompr_idx is None:
        print("Preparing lattice translation (Sparse)")
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_sparse_O3(
            trans_perms, fc_cutoff
        )
    nonzero = np.where(atomic_decompr_idx != -1)[0]

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        """Equivalent to mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
        C: atomic_lat_trans_compr_mat, shape=(NNN, NNN/n_lp)"""
        print(
            "Coset sum:", str(i + 1) + "/" + str(len(spg_reps.unique_rotation_indices))
        )
        permutation = spg_reps.get_sigma3_rep_vec(i)
        atomic_decompr_idx_permuted = atomic_decompr_idx[permutation]
        nonzero = nonzero[atomic_decompr_idx_permuted[nonzero] != -1]

        mat = csr_array(
            (
                np.ones(len(nonzero), dtype="int_"),
                (atomic_decompr_idx_permuted[nonzero], atomic_decompr_idx[nonzero]),
            ),
            shape=(N**3 // n_lp, N**3 // n_lp),
            dtype="int_",
        )
        mat = kron(mat, spg_reps.r_reps[i] * factor)
        if c_pt is not None:
            mat = c_pt.T @ mat @ c_pt

        coset_reps_sum += mat

    return coset_reps_sum
