#!/usr/bin/env python
import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice
from sysfc.utils.utils_O3 import get_lat_trans_decompr_indices_O3


def compressed_complement_projector_sum_rules_from_compact_compr_mat(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Calculate a complementary projector for sum rules.

    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans. Batch calculations are used to reduce
    the amount of memory allocation.

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


def compressed_projector_sum_rules_from_compact_compr_mat(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule.

    This is compressed by C_compr = C_trans @ n_a_compress_mat.
    """
    proj_cplmt = compressed_complement_projector_sum_rules_from_compact_compr_mat(
        trans_perms, n_a_compress_mat, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
