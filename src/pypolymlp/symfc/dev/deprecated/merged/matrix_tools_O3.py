#!/usr/bin/env python
import numpy as np
from scipy.sparse import csr_array
from symfc.utils.cutoff_tools_O3 import FCCutoffO3
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools_O3 import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def N3N3N3_to_NNNand333(combinations_perm: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNNN = combinations_perm[:, 0] // 3 * N**2
    vecNNN += combinations_perm[:, 1] // 3 * N
    vecNNN += combinations_perm[:, 2] // 3
    vec333 = combinations_perm[:, 0] % 3 * 9
    vec333 += combinations_perm[:, 1] % 3 * 3
    vec333 += combinations_perm[:, 2] % 3
    return vecNNN, vec333


def projector_permutation_lat_trans(
    trans_perms,
    fc_cutoff: FCCutoffO3 = None,
    n_batch=12,
    use_mkl=False,
):
    """Calculate a projector for permutation rules compressed by C_trans
    without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.
    fc_cutoff : FCCutoffO3

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms) * 27

    # (1) for FC3 with single index ia
    if fc_cutoff is None:
        combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    else:
        combinations = fc_cutoff.combinations1()

    n_perm1 = combinations.shape[0]
    combinations, combinations333 = N3N3N3_to_NNNand333(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm1, 1.0 / np.sqrt(n_lp)),
            (np.arange(n_perm1), decompr_idx[combinations] + combinations333),
        ),
        shape=(n_perm1, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    if fc_cutoff is None:
        combinations = get_combinations(3 * natom, 2)
    else:
        combinations = fc_cutoff.combinations2()

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

    c_pt = csr_array(
        (
            np.full(n_perm2 * 3, 1 / np.sqrt(3 * n_lp)),
            (
                np.repeat(range(n_perm2), 3),
                decompr_idx[combinations] + combinations333,
            ),
        ),
        shape=(n_perm2, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    """Bottleneck part for memory reduction in constructing a basis set.
    Moreover, combinations can be divided using fc_cut.combiations3(i).
    """
    if fc_cutoff is None:
        combinations = get_combinations(3 * natom, 3)
    else:
        combinations = fc_cutoff.combinations3_all()

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
        print("Proj (perm.T @ trans):", str(end) + "/" + str(n_perm3))
        batch_size = end - begin
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 3))
        combinations_perm, combinations333 = N3N3N3_to_NNNand333(
            combinations_perm, natom
        )

        c_pt = csr_array(
            (
                np.full(batch_size * 6, 1 / np.sqrt(6 * n_lp)),
                (
                    np.repeat(np.arange(batch_size), 6),
                    decompr_idx[combinations_perm] + combinations333,
                ),
            ),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt
