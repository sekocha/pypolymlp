#!/usr/bin/env python
import gc
import math

import numpy as np
from scipy.sparse import csr_array
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools_O3 import N3N3N3_to_NNN333, get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import (
    get_atomic_lat_trans_decompr_indices_O3,
    get_lat_trans_decompr_indices_O3,
)


def N3N3N3_to_NNNand333(combinations_perm: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    # vecNNN = combinations_perm[:, 0] // 3 * 27 * N**2
    # vecNNN += combinations_perm[:, 1] // 3 * 27 * N
    # vecNNN += combinations_perm[:, 2] // 3 * 27
    vecNNN = combinations_perm[:, 0] // 3 * N**2
    vecNNN += combinations_perm[:, 1] // 3 * N
    vecNNN += combinations_perm[:, 2] // 3
    vec333 = combinations_perm[:, 0] % 3 * 9
    vec333 += combinations_perm[:, 1] % 3 * 3
    vec333 += combinations_perm[:, 2] % 3
    return vecNNN, vec333


def apply_zeros(c_pt, decompr_idx, zero_ids):
    """
    Parameters
    ----------
    zero_ids: Zero elements in NNN333 format.

    Slow but simple implementation to apply zero elements to C_pt = C_perm.T @ C_trans.
    This function is not used elsewhere.
    """
    for i in zero_ids:
        nonzero_rows = c_pt.getcol(decompr_idx[i]).nonzero()[0]
        for j in nonzero_rows:
            c_pt[j, decompr_idx[i]] = 0
    return c_pt


def projector_permutation_lat_trans_sparse(
    trans_perms, n_batch=12, use_mkl=False, zero_ids=None
):
    """Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    """Bottleneck part for memory reduction in constructing a basis set.
    Input zero elements must be applied to decompr_idx.
    """
    decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    # (1) for FC3 with single index ia
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    n_perm1 = combinations.shape[0]
    combinations, combinations333 = N3N3N3_to_NNNand333(combinations, natom)

    data = np.full(n_perm1, 1.0 / math.sqrt(n_lp))
    if zero_ids is not None:
        """This part will be more efficient by revising using atomic_decompr_idx."""
        data[np.isin(combinations * 27 + combinations333, zero_ids)] = 0

    c_pt = csr_array(
        (data, (np.arange(n_perm1), decompr_idx[combinations] * 27 + combinations333)),
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
    combinations, combinations333 = N3N3N3_to_NNNand333(combinations, natom)

    data = np.full(n_perm2 * 3, 1 / math.sqrt(3 * n_lp))
    if zero_ids is not None:
        data[np.isin(combinations * 27 + combinations333, zero_ids)] = 0

    c_pt = csr_array(
        (
            data,
            (
                np.repeat(range(n_perm2), 3),
                decompr_idx[combinations] * 27 + combinations333,
            ),
        ),
        shape=(n_perm2, NNN27 // n_lp),
        dtype="double",
    )

    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    """Bottleneck part for memory reduction in constructing a basis set.
    Input zero elements must be applied to decompr_idx.
    Moreover, it is better that combinations can be divided.
    """
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
        combinations_perm, combinations333 = N3N3N3_to_NNNand333(
            combinations_perm, natom
        )

        data = np.full(batch_size * 6, 1 / math.sqrt(6 * n_lp))
        if zero_ids is not None:
            data[np.isin(combinations_perm * 27 + combinations333, zero_ids)] = 0

        c_pt = csr_array(
            (
                data,
                (
                    np.repeat(np.arange(batch_size), 6),
                    decompr_idx[combinations_perm] * 27 + combinations333,
                ),
            ),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )
        del data, combinations_perm, combinations333
        gc.collect()

        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

        del c_pt
        gc.collect()

    return proj_pt


def projector_permutation_lat_trans(
    trans_perms, n_batch=12, use_mkl=False, zero_ids=None
):
    """Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    """Bottleneck part for memory reduction in constructing a basis set.
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
        (data, (np.arange(n_perm1), decompr_idx[combinations])),
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

    data = np.full(n_perm2 * 3, 1 / math.sqrt(3 * n_lp))
    if zero_ids is not None:
        data[np.isin(combinations, zero_ids)] = 0

    c_pt = csr_array(
        (data, (np.repeat(range(n_perm2), 3), decompr_idx[combinations])),
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

        data = np.full(batch_size * 6, 1 / math.sqrt(6 * n_lp))
        if zero_ids is not None:
            data[np.isin(combinations_perm, zero_ids)] = 0

        c_pt = csr_array(
            (
                data,
                (
                    np.repeat(np.arange(batch_size), 6),
                    decompr_idx[combinations_perm],
                ),
            ),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )

        del data
        gc.collect()

        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

        del c_pt
        gc.collect()

    return proj_pt
