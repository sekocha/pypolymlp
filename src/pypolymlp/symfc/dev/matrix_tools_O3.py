#!/usr/bin/env python 
import numpy as np
import math

from scipy.sparse import csr_array

from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O3 import get_lat_trans_decompr_indices_O3
from symfc.utils.matrix_tools_O3 import N3N3N3_to_NNN333, get_combinations
from symfc.utils.solver_funcs import get_batch_slice

def apply_zeros(c_pt, decompr_idx, zero_ids):
    """
    Parameters
    ----------
    zero_ids: Zero elements in NNN333 format.
    """
    for i in zero_ids:
        print(i)
        nonzero_rows = c_pt.getcol(decompr_idx[i]).nonzero()[0]
        for j in nonzero_rows:
            c_pt[j, decompr_idx[i]] = 0
    return c_pt


def projector_permutation_lat_trans(
    trans_perms, n_batch=6, use_mkl=False, zero_ids=None
):
    '''Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm. 
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans 
    '''
    n_lp, natom = trans_perms.shape
    """Bottleneck part for memory reduction. 
    Input zero elements must be applied to decompr_idx.
    """
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    NNN27 = natom**3 * 27

    # (1) for FC3 with single index ia
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    n_perm1 = combinations.shape[0]
    combinations = N3N3N3_to_NNN333(combinations, natom)

    data = np.full(n_perm1, 1.0 / math.sqrt(n_lp))
    if zero_ids is not None:
        data[np.isin(combinations, zero_ids)] = 0

    c_pt = csr_array(
        (
            data, (np.arange(n_perm1), decompr_idx[combinations])
        ),
        shape=(n_perm1, NNN27 // n_lp), dtype="double",
    )

    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)


    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    #combinations = np.stack(np.triu_indices(3 * natom, k=1), axis=-1)
    combinations = get_combinations(3*natom, 2)
    n_perm2 = combinations.shape[0] * 2

    perms = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    combinations = combinations[:, perms].reshape((-1, 3))
    combinations = N3N3N3_to_NNN333(combinations, natom)

    data = np.full(n_perm2 * 3, 1 / math.sqrt(3 * n_lp))
    if zero_ids is not None:
        data[np.isin(combinations, zero_ids)] = 0

    c_pt = csr_array(
        (
            data, (np.repeat(range(n_perm2), 3), decompr_idx[combinations])
        ),
        shape=(n_perm2, NNN27 // n_lp), dtype="double",
    )

    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)


    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    combinations = get_combinations(3*natom, 3)
    n_perm3 = combinations.shape[0]

    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for begin, end in zip(*get_batch_slice(n_perm3, n_perm3 // n_batch)):
        print('Proj (perm.T @ trans) batch:', end)
        batch_size = end - begin
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 3))
        combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

        data = np.full(batch_size * 6, 1 / math.sqrt(6 * n_lp))
        if zero_ids is not None:
            data[np.isin(combinations_perm, zero_ids)] = 0

        c_pt = csr_array(
            (
                data, 
                (np.repeat(np.arange(batch_size), 6), 
                    decompr_idx[combinations_perm])
            ),
            shape=(batch_size, NNN27 // n_lp), dtype="double",
        )

        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt


