#!/usr/bin/env python 
import numpy as np
import itertools
import math
import gc

from scipy.sparse import csr_array

from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O3 import get_lat_trans_decompr_indices_O3
from symfc.utils.matrix_tools_O3 import N3N3N3_to_NNN333
from symfc.solvers.solver_funcs import get_batch_slice

import scipy


def compressed_complement_projector_sum_rules_lat_trans(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    '''Calculate a complementary projector for sum rules compressed by C_trans
    without allocating C_trans. 
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector 
    P_sum_(c) = C_trans.T @ C_sum_(c) @ C_sum_(c).T @ C_trans 
    '''

    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NN27 = natom**2 * 27

    proj_size = n_a_compress_mat.shape[1]
    proj_sum_cplmt = csr_array(
            ([], ([], [])), shape=(proj_size, proj_size), dtype="double",
            )

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((natom,NN27)).T.reshape(-1)

    n_batch = 9
    if n_batch == 3:
        batch_size_vector = natom**3*9
        batch_size_matrix = natom**2*9
    elif n_batch == 9:
        batch_size_vector = natom**3*3
        batch_size_matrix = natom**2*3
    elif n_batch == 27:
        batch_size_vector = natom**3
        batch_size_matrix = natom**2
    elif n_batch == natom:
        batch_size_vector = natom**2*27
        batch_size_matrix = natom*27
    else:
        raise ValueError('n_batch = 9, 27, or N.')

    for begin, end in zip(*get_batch_slice(NNN27, batch_size_vector)):
        print('Proj_complement (sum.T @ trans) batch:', end)
        batch_size = end - begin
        c_sum_cplmt = csr_array(
            (
                np.ones(batch_size_vector, dtype="double"), 
                (np.repeat(np.arange(batch_size_matrix), natom), 
                 decompr_idx[begin:end]),
            ),
            shape=(batch_size_matrix, NNN27 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(
            c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl
        )
        proj_sum_cplmt += dot_product_sparse(
            c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
        )

    proj_sum_cplmt /= (n_lp * natom)
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    """Return complementary projection matrix for sum rule compressed by C."""
    return compressed_complement_projector_sum_rules_lat_trans(
        n_a_compress_mat, trans_perms, use_mkl=use_mkl
    )


def compressed_projector_sum_rules(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = compressed_complement_projector_sum_rules(
        n_a_compress_mat, trans_perms, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def projector_permutation_lat_trans(trans_perms, n_batch=6, use_mkl=False):
    '''Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm. 
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans 
    '''

    n_lp, natom = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    NNN27 = natom**3 * 27

    combinations3 = np.array(
        list(itertools.combinations(range(3 * natom), 3)), dtype=int
    )
    combinations2 = np.array(
        list(itertools.combinations(range(3 * natom), 2)), dtype=int
    )
    combinations1 = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)

    n_perm3 = combinations3.shape[0]
    n_perm2 = combinations2.shape[0] * 2
    n_perm1 = combinations1.shape[0]
    n_perm = n_perm3 + n_perm2 + n_perm1
    print('Number of FC3 combinations =', n_perm)

    proj_pt = csr_array(
            ([], ([],[])), 
            shape=(NNN27 // n_lp, NNN27 // n_lp), 
            dtype="double",
            )

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for begin, end in zip(*get_batch_slice(n_perm3, n_perm3 // n_batch)):
        print('Proj (perm.T @ trans) batch:', end)
        batch_size = end - begin
        combinations_perm = combinations3[begin:end][:, perms].reshape((-1, 3))
        combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

        c_pt = csr_array(
            (np.full(batch_size * 6, 1 / math.sqrt(6 * n_lp)), 
            (np.repeat(np.arange(batch_size), 6), 
                decompr_idx[combinations_perm])),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    del combinations3
    gc.collect()

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    perms = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    combinations2 = combinations2[:, perms].reshape((-1, 3))
    combinations2 = N3N3N3_to_NNN333(combinations2, natom)

    c_pt = csr_array(
        (np.full(n_perm2 * 3, 1 / math.sqrt(3 * n_lp)), 
        (np.repeat(range(n_perm2), 3), decompr_idx[combinations2])),
        shape=(n_perm, NNN27 // n_lp), dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (1) for FC3 with single index ia
    combinations1 = N3N3N3_to_NNN333(combinations1, natom)
    c_pt = csr_array(
        (np.full(n_perm1, 1.0 / math.sqrt(n_lp)), 
        (np.arange(n_perm1), decompr_idx[combinations1])),
        shape=(n_perm, NNN27 // n_lp), dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt


