import numpy as np
from scipy.sparse import csr_array, kron

from symfc.spg_reps import SpgRepsO3
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def get_compr_coset_reps_sum_O3(spg_reps: SpgRepsO3) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**3 * 27 // n_lp
    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    C = csr_array(
        (
            np.ones(N**3, dtype=int),
            (np.arange(N**3, dtype=int), atomic_decompr_idx),
        ),
        shape=(N**3, N**3 // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma3_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)

    return coset_reps_sum


