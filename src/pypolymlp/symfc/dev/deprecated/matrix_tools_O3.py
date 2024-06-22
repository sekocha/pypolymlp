#!/usr/bin/env python
import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def compressed_projector_sum_rules_O3(
    trans_perms,
    n_a_compress_mat: csr_array,
    fc_cutoff=None,
    atomic_decompr_idx=None,
    use_mkl: bool = False,
    n_batch=None,
) -> csr_array:
    """Return projection matrix for sum rule.

    Calculate a complementary projector for sum rules.
    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans.
    Memory efficient version using get_atomic_lat_trans_decompr_indices_O3.

    Return
    ------
    Compressed projector I - P^(c)
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NNN = natom**3
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if n_batch is None:
        if natom < 256:
            n_batch = natom // min(natom, 16)
        else:
            n_batch = natom // 4

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**2 * (natom // n_batch)

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NN)).T.reshape(-1) * 27
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices()
        nonzero = nonzero.reshape((natom, NN)).T.reshape(-1)

    abc = np.arange(27)
    for begin, end in zip(*get_batch_slice(NNN, batch_size)):
        print("Complementary P (Sum rule):", str(end) + "/" + str(NNN))
        size = end - begin
        size_vector = size * 27
        size_row = size_vector // natom

        if fc_cutoff is None:
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_vector, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom),
                        (decompr_idx[begin:end][None, :] + abc[:, None]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            size_data = np.count_nonzero(nonzero_b) * 27
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 27)],
                        (
                            decompr_idx[begin:end][nonzero_b][None, :] + abc[:, None]
                        ).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
