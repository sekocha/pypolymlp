import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools_O3 import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def N3N3_to_NNand33(combinations_perm: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNN, vec33 = np.divmod(combinations_perm, 3)
    vecNN = vecNN @ np.array([N, 1])
    vec33 = vec33 @ np.array([3, 1])
    return vecNN, vec33


def projector_permutation_lat_trans_O2(
    trans_perms: np.ndarray,
    atomic_decompr_idx: np.ndarray = None,
    fc_cutoff: FCCutoff = None,
    use_mkl=False,
):
    """Calculate a projector for permutation rules compressed by C_trans.

    This is calculated without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.
    fc_cutoff : FCCutoff

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    # (1) for FC2 with single index ia
    combinations = np.array([[i, i] for i in range(3 * natom)], dtype=int)
    n_perm1 = combinations.shape[0]
    combinations, combinations33 = N3N3_to_NNand33(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm1, 1.0 / np.sqrt(n_lp)),
            (
                np.arange(n_perm1),
                atomic_decompr_idx[combinations] * 9 + combinations33,
            ),
        ),
        shape=(n_perm1, NN9 // n_lp),
        dtype="double",
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (2) for FC2 with two distinguished indices (ia,jb)
    if fc_cutoff is None:
        combinations = get_combinations(3 * natom, 2)
    else:
        combinations = fc_cutoff.combinations2()

    n_perm2 = combinations.shape[0]
    perms = [
        [0, 1],
        [1, 0],
    ]
    combinations = combinations[:, perms].reshape((-1, 2))
    combinations, combinations33 = N3N3_to_NNand33(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm2 * 2, 1 / np.sqrt(2 * n_lp)),
            (
                np.repeat(range(n_perm2), 2),
                atomic_decompr_idx[combinations] * 9 + combinations33,
            ),
        ),
        shape=(n_perm2, NN9 // n_lp),
        dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt


def compressed_projector_sum_rules_O2(
    trans_perms,
    n_a_compress_mat: csr_array,
    fc_cutoff=None,
    atomic_decompr_idx=None,
    use_mkl: bool = False,
    n_batch=1,
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
    NN9 = natom**2 * 9
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom * (natom // n_batch)

    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, natom)).T.reshape(-1) * 9
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc2()
        nonzero = nonzero.reshape((natom, natom)).T.reshape(-1)

    ab = np.arange(9)
    for begin, end in zip(*get_batch_slice(NN, batch_size)):
        size = end - begin
        size_vector = size * 9
        size_row = size_vector // natom

        if fc_cutoff is None:
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_vector, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom),
                        (decompr_idx[begin:end][None, :] + ab[:, None]).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            size_data = np.count_nonzero(nonzero_b) * 9
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 9)],
                        (
                            decompr_idx[begin:end][nonzero_b][None, :] + ab[:, None]
                        ).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
