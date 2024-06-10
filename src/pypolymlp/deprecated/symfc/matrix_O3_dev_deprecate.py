#!/usr/bin/env python
import gc
import itertools
import math

import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.basis_sets.basis_sets_O3 import print_sp_matrix_size
from symfc.solvers.solver_funcs import get_batch_slice
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools_O3 import N3N3N3_to_NNN333, get_perm_compr_matrix_O3
from symfc.utils.utils_O3 import (
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)
from utils_O3_dev import get_compr_coset_reps_sum_O3


def compressed_complement_projector_sum_rules_lat_trans_stable(
    n_a_compress_mat: csr_array, trans_perms, use_mkl: bool = False
) -> csr_array:
    """
    Return
    ------
    C_trans.T @ C_sum_cplmt @ C_sum_cplmt @ C_trans

    Calculate complementary projector for sum rules compressed by C_trans.
    This version is easier to read than a more memory-efficient version.
    """

    n_lp, N = trans_perms.shape
    NNN27 = N**3 * 27
    NN27 = N**2 * 27

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((N, NN27)).T.reshape(-1)
    row = np.repeat(np.arange(NN27), N)
    c_sum_cplmt = csr_array(
        (
            np.ones(NNN27, dtype="double"),
            (row, decompr_idx),
        ),
        shape=(NN27, NNN27 // n_lp),
        dtype="double",
    )
    c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
    proj_sum_cplmt = dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)
    proj_sum_cplmt /= n_lp * N

    return proj_sum_cplmt


def permutation_dot_lat_trans_stable(trans_perms):
    """
    Retern
    ------
    C_perm.T @ C_trans

    Calculate C_pt = C_perm.T @ C_trans using C_trans and C_perm.
    """

    n_lp, N = trans_perms.shape
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    print_sp_matrix_size(c_trans, " C_(trans):")

    c_perm = get_perm_compr_matrix_O3(N)
    print_sp_matrix_size(c_perm, " C_(perm):")

    c_pt = c_perm.T @ c_trans
    return c_pt


def permutation_dot_lat_trans(trans_perms):
    """
    Retern
    ------
    C_perm.T @ C_trans

    Calculate C_pt = C_perm.T @ C_trans without allocating C_trans and C_perm.
    """

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

    n_data3 = combinations3.shape[0] * 6
    n_data2 = combinations2.shape[0] * 6
    n_data1 = combinations1.shape[0]
    n_data = n_data3 + n_data2 + n_data1

    row = np.zeros(n_data, dtype="int_")
    col = np.zeros(n_data, dtype="int_")
    data = np.zeros(n_data, dtype="double")

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    begin_id, end_id = 0, n_data3
    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    combinations_perm = combinations3[:, perms].reshape((-1, 3))
    combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

    row[begin_id:end_id] = np.repeat(range(n_perm3), 6)
    col[begin_id:end_id] = decompr_idx[combinations_perm]
    data[begin_id:end_id] = 1 / math.sqrt(6 * n_lp)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    begin_id = end_id
    end_id = begin_id + n_data2
    perms = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    combinations_perm = combinations2[:, perms].reshape((-1, 3))
    combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

    row[begin_id:end_id] = np.repeat(range(n_perm3, n_perm3 + n_perm2), 3)
    col[begin_id:end_id] = decompr_idx[combinations_perm]
    data[begin_id:end_id] = 1 / math.sqrt(3 * n_lp)

    # (1) for FC3 with single index ia
    begin_id = end_id
    combinations_perm = N3N3N3_to_NNN333(combinations1, natom)
    row[begin_id:] = np.array(range(n_perm3 + n_perm2, n_perm))
    col[begin_id:] = decompr_idx[combinations_perm]
    data[begin_id:] = 1.0 / math.sqrt(n_lp)

    c_pt = csr_array(
        (data, (row, col)),
        shape=(n_perm, NNN27 // n_lp),
        dtype="double",
    )
    return c_pt
