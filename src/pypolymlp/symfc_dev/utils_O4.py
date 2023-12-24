#!/usr/bin/env python
import numpy as np
from scipy.sparse import coo_array, kron

from symfc.utils import get_indep_atoms_by_lat_trans
from pypolymlp.symfc_dev.spg_reps_O4 import SpgRepsO4

def get_atomic_lat_trans_decompr_indices_O4(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by atom-lat-trans-sym.

    This is atomic permutation only version of get_lat_trans_decompr_indices.

    Usage
    -----
    vec[indices] of shape (n_a*N*N*N,) gives an array of shape=(N**4,).
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
        Indices of n_a * N * N * N elements.
        shape=(N^4*,), dtype='int_'.

    """

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N**4

    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**3
        for j in range(N):
            index_shift_j = index_shift_i + trans_perms[:, j] * N**2
            for k in range(N):
                index_shift_k = index_shift_j + trans_perms[:, k] * N
                for l in range(N):
                    index_shift = index_shift_k + trans_perms[:, l]
                    indices[index_shift] = n
                    n += 1
    assert n * n_lp == size_row
    return indices

def get_lat_trans_compr_matrix_O4(decompr_idx: np.ndarray, N: int, n_lp: int) -> coo_array:
    """Return compression matrix by lattice translation symmetry.

    `decompr_idx` is obtained by `get_lat_trans_decompr_indices`.

    Matrix shape is (NNNN333, n_a*NNN333), where n_a is the number of independent
    atoms by lattice translation symmetry.

    Data order is (N, N, N, N, 3, 3, 3, 3, n_a, N, N, N, 3, 3, 3, 3) 
    if it is in dense array.

    """
    NNNN81 = N**4 * 81
    compression_mat = coo_array(
        (
            np.full(NNNN81, 1 / np.sqrt(n_lp), dtype="double"),
            (np.arange(NNNN81, dtype=int), decompr_idx),
        ),
        shape=(NNNN81, NNNN81 // n_lp),
        dtype="double",
    )
    return compression_mat

def get_lat_trans_decompr_indices_O4(trans_perms: np.ndarray) -> np.ndarray:
    """Return indices to de-compress compressed matrix by lat-trans-sym.

    Usage
    -----
    vec[indices] of shape (n_a*N*N*N*81,) gives an array of shape=(N**4*81,).
    1/sqrt(n_lp) must be multiplied manually after decompression to mimic
    get_lat_trans_compr_matrix.

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
        Indices of n_a * N * N * N * 81 elements.
        shape=(N^4*81,), dtype='int_'.

    """
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_a = len(indep_atoms)
    N = trans_perms.shape[1]
    n_lp = N // n_a
    size_row = 81 * N**4

    trans_perms = trans_perms.astype("int_")
    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift_i = trans_perms[:, i_patom] * N**3 * 81
        for j in range(N):
            index_shift_j = index_shift_i + trans_perms[:, j] * N**2 * 81
            for k in range(N):
                index_shift_k = index_shift_j + trans_perms[:, k] * N * 81
                for l in range(N):
                    index_shift = index_shift_k + trans_perms[:, l] * 81
                    for abcd in range(81):
                        indices[index_shift + abcd] = n
                        n += 1
    assert n * n_lp == size_row
    return indices

def get_compr_coset_reps_sum_O4(spg_reps: SpgRepsO4):

    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**4 * 81 // n_lp
    coset_reps_sum = coo_array(([], ([], [])), 
                                shape=(size, size), 
                                dtype="double")
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    C = coo_array(
        (
            np.ones(N**4, dtype=int),
            (np.arange(N**4, dtype=int), atomic_decompr_idx),
        ),
        shape=(N**4, N**4 // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma4_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)
    return coset_reps_sum


