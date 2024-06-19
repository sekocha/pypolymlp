"""Utility functions for 3rd order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron
from symfc.spg_reps import SpgRepsO3
from symfc.utils.cutoff_tools_O3 import FCCutoffO3
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def get_compr_coset_reps_sum_O3_sparse(
    spg_reps: SpgRepsO3,
    fc_cutoff: FCCutoffO3,
    atomic_decompr_idx=None,
    c_pt: csr_array = None,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**3 * 27 // n_lp if c_pt is None else c_pt.shape[1]
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    nonzero = fc_cutoff.nonzero_atomic_indices()
    size_nonzero = np.count_nonzero(nonzero)

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        """Equivalent to mat = C.T @ spg_reps.get_sigma3_rep(i) @ C
        C: atomic_lat_trans_compr_mat, shape=(NNN, NNN/n_lp)"""
        print(
            "Coset sum:", str(i + 1) + "/" + str(len(spg_reps.unique_rotation_indices))
        )
        permutation = spg_reps.get_sigma3_rep_vec(i)[nonzero]
        mat = csr_array(
            (
                np.ones(size_nonzero, dtype="int_"),
                (atomic_decompr_idx[permutation], atomic_decompr_idx[nonzero]),
            ),
            shape=(N**3 // n_lp, N**3 // n_lp),
            dtype="int_",
        )
        mat = kron(mat, spg_reps.r_reps[i] * factor)
        if c_pt is not None:
            mat = c_pt.T @ mat @ c_pt

        coset_reps_sum += mat

    return coset_reps_sum
