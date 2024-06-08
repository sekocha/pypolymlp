import math
import time

import numpy as np
from scipy.sparse import csr_array, kron
from symfc.solvers.solver_funcs import get_batch_slice
from symfc.spg_reps import SpgRepsO3
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def get_compr_coset_reps_sum_O3(
    spg_reps: SpgRepsO3, use_mkl: bool = False
) -> csr_array:
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

    n_coset_rep = len(spg_reps.unique_rotation_indices)
    n_batch = round(math.sqrt(n_coset_rep))
    batch_size = n_coset_rep // n_batch

    for begin, end in zip(*get_batch_slice(n_coset_rep, batch_size)):
        coset_reps_sum_partial = csr_array(
            ([], ([], [])), shape=(size, size), dtype="double"
        )
        print("Coset batch_size:", end)
        for i in range(begin, end):
            mat = spg_reps.get_sigma3_rep(i)
            mat = mat @ C
            mat = C.T @ mat
            coset_reps_sum_partial += kron(mat, spg_reps.r_reps[i])
        coset_reps_sum += coset_reps_sum_partial

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    coset_reps_sum *= factor
    return coset_reps_sum
