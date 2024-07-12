#!/usr/bin/env python
import gc
import itertools
import math
import time

import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.solvers.solver_funcs import get_batch_slice
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools_O3 import N3N3N3_to_NNN333
from symfc.utils.utils_O3 import get_lat_trans_decompr_indices_O3


def set_complement_sum_rules_lat_trans(trans_perms) -> csr_array:
    """Calculate a decomposition of complementary projector
    for sum rules compressed by C_trans without allocating C_trans.

    P_sum^(c) = C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
              = [C_sum^(c).T @ C_trans].T @ [C_sum^(c).T @ C_trans]

    Return
    ------
    Product of C_sum^(c).T and C_trans.
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NN27 = natom**2 * 27

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((natom, NN27)).T.reshape(-1)

    row = np.repeat(np.arange(NN27), natom)
    c_sum_cplmt = csr_array(
        (
            np.ones(NNN27, dtype="double"),
            (row, decompr_idx),
        ),
        shape=(NN27, NNN27 // n_lp),
        dtype="double",
    )
    c_sum_cplmt /= n_lp * natom
    return c_sum_cplmt


def compressed_complement_projector_sum_rules_lat_trans(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Calculate a complementary projector for sum rules compressed by
    C_trans and n_a_compress_mat without allocating C_trans.
    Batch calculations are used to reduce the amount of memory allocation.

    Return
    ------
    Compressed projector
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """

    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NN27 = natom**2 * 27

    proj_size = n_a_compress_mat.shape[1]
    proj_sum_cplmt = csr_array(
        ([], ([], [])),
        shape=(proj_size, proj_size),
        dtype="double",
    )

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((natom, NN27)).T.reshape(-1)

    n_batch = 9
    if n_batch == 3:
        batch_size_vector = natom**3 * 9
        batch_size_matrix = natom**2 * 9
    elif n_batch == 9:
        batch_size_vector = natom**3 * 3
        batch_size_matrix = natom**2 * 3
    elif n_batch == 27:
        batch_size_vector = natom**3
        batch_size_matrix = natom**2
    elif n_batch == natom:
        batch_size_vector = natom**2 * 27
        batch_size_matrix = natom * 27
    else:
        raise ValueError("n_batch = 9, 27, or N.")

    for begin, end in zip(*get_batch_slice(NNN27, batch_size_vector)):
        print("Proj_complement (sum.T @ trans) batch:", end)
        batch_size = end - begin
        c_sum_cplmt = csr_array(
            (
                np.ones(batch_size_vector, dtype="double"),
                (
                    np.repeat(np.arange(batch_size_matrix), natom),
                    decompr_idx[begin:end],
                ),
            ),
            shape=(batch_size_matrix, NNN27 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_sum_cplmt += dot_product_sparse(
            c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
        )

    proj_sum_cplmt /= n_lp * natom
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules(
    trans_perms, n_a_compress_mat: csr_array = None, use_mkl: bool = False
) -> csr_array:
    """Return complementary projection matrix for sum rule compressed by
    C_compr = C_trans @ n_a_compress_mat."""
    return compressed_complement_projector_sum_rules_lat_trans(
        trans_perms, n_a_compress_mat, use_mkl=use_mkl
    )


def compressed_projector_sum_rules(
    trans_perms, n_a_compress_mat: csr_array = None, use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule compressed by
    C_compr = C_trans @ n_a_compress_mat."""
    proj_cplmt = compressed_complement_projector_sum_rules(
        trans_perms, n_a_compress_mat, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def get_combinations(n, r):
    """
    combinations = np.array(
        list(itertools.combinations(range(n), r)), dtype=int
    )
    """
    combs = np.ones((r, n - r + 1), dtype=int)
    combs[0] = np.arange(n - r + 1)
    for j in range(1, r):
        reps = (n - r + j) - combs[j - 1]
        combs = np.repeat(combs, reps, axis=1)
        ind = np.add.accumulate(reps)
        combs[j, ind[:-1]] = 1 - reps[1:]
        combs[j, 0] = j
        combs[j] = np.add.accumulate(combs[j])
    return combs.T


def projector_permutation_lat_trans(trans_perms, n_batch=6, use_mkl=False):
    """Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    NNN27 = natom**3 * 27

    # (1) for FC3 with single index ia
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    n_perm1 = combinations.shape[0]

    combinations = N3N3N3_to_NNN333(combinations, natom)
    c_pt = csr_array(
        (
            np.full(n_perm1, 1.0 / math.sqrt(n_lp)),
            (np.arange(n_perm1), decompr_idx[combinations]),
        ),
        shape=(n_perm1, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    # combinations = np.stack(np.triu_indices(3 * natom, k=1), axis=-1)
    combinations = get_combinations(3 * natom, 2)
    n_perm2 = combinations.shape[0] * 2

    perms = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    combinations = combinations[:, perms].reshape((-1, 3))
    combinations = N3N3N3_to_NNN333(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm2 * 3, 1 / math.sqrt(3 * n_lp)),
            (np.repeat(range(n_perm2), 3), decompr_idx[combinations]),
        ),
        shape=(n_perm2, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    combinations = get_combinations(3 * natom, 3)
    n_perm3 = combinations.shape[0]

    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    for begin, end in zip(*get_batch_slice(n_perm3, n_perm3 // n_batch)):
        print("Proj (perm.T @ trans) batch:", end)
        batch_size = end - begin
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 3))
        combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

        c_pt = csr_array(
            (
                np.full(batch_size * 6, 1 / math.sqrt(6 * n_lp)),
                (
                    np.repeat(np.arange(batch_size), 6),
                    decompr_idx[combinations_perm],
                ),
            ),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt
