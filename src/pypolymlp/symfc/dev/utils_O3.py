#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_array, kron
from symfc.utils.utils import get_indep_atoms_by_lat_trans

from pypolymlp.symfc.dev.zero_tools_O3 import FCCutoffO3

# import time


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


def get_compr_coset_reps_sum_sparse_O3(spg_reps, fc_cutoff, c_pt, use_mkl=False):
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = c_pt.shape[1]
    proj_rpt = csr_array(([], ([], [])), shape=(size, size), dtype="double")

    """Todo: better interface"""
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_sparse_O3(
        trans_perms, fc_cutoff
    )
    match = np.where(atomic_decompr_idx != -1)[0]
    C = csr_array(
        (np.ones(len(match), dtype=int), (match, atomic_decompr_idx[match])),
        shape=(N**3, N**3 // n_lp),
    )

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        print(
            "Coset sum:", str(i + 1) + "/" + str(len(spg_reps.unique_rotation_indices))
        )
        #    t1 = time.time()
        """Bottleneck part"""
        # proj_pt = dot_product_sparse(
        #     dot_product_sparse(
        #         C.T, spg_reps.get_sigma3_rep(i), use_mkl=use_mkl
        #     ), C, use_mkl=use_mkl
        # )
        mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
        #    t2 = time.time()
        kron1 = kron(mat, spg_reps.r_reps[i] * factor)
        #    t3 = time.time()
        proj_rpt += c_pt.T @ kron1 @ c_pt
    #    t4 = time.time()
    #    print(t2-t1, t3-t2, t4-t3)

    return proj_rpt
