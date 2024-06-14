#!/usr/bin/env python
import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import (  # get_lat_trans_decompr_indices_O3,
    get_atomic_lat_trans_decompr_indices_O3,
)


def compressed_complement_projector_sum_rules_from_compact_compr_mat(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Calculate a complementary projector for sum rules.

    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans.
    Memory efficient version using get_atomic_lat_trans_decompr_indices_O3.

    Return
    ------
    Compressed projector
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NNN = natom**3
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_sum_cplmt = csr_array(
        ([], ([], [])),
        shape=(proj_size, proj_size),
        dtype="double",
    )

    decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms) * 27
    decompr_idx = decompr_idx.reshape((natom, NN)).T.reshape(-1)

    n_batch = natom
    if n_batch == natom:
        batch_size_vector = natom**2
        batch_size_matrix = natom * 27
    else:
        raise ValueError("n_batch must be N.")

    for begin, end in zip(*get_batch_slice(NNN, batch_size_vector)):
        print("Proj_complement (sum.T @ trans) batch:", str(end) + "/" + str(NNN))
        col = np.add.outer(np.arange(27), decompr_idx[begin:end]).reshape(-1)
        c_sum_cplmt = csr_array(
            (
                np.ones(batch_size_vector * 27, dtype="double"),
                (
                    np.repeat(np.arange(batch_size_matrix), natom),
                    col,
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
