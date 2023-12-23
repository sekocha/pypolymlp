#!/usr/bin/env python
import numpy as np
from scipy.sparse import coo_array, kron

from symfc.utils import get_indep_atoms_by_lat_trans
from symfc.spg_reps import SpgRepsO1

def get_atomic_lat_trans_decompr_indices_O1(trans_perms: np.ndarray) -> np.ndarray:

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    n_lp, N = trans_perms.shape
    size_row = N

    n = 0
    indices = np.zeros(size_row, dtype="int_")
    for i_patom in indep_atoms:
        index_shift = trans_perms[:, i_patom]
        indices[index_shift] = n
        n += 1
    assert n * n_lp == size_row
    return indices



def get_compr_coset_reps_sum_O1(spg_reps: SpgRepsO1):

    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N * 3 // n_lp
    coset_reps_sum = coo_array(([], ([], [])), 
                                shape=(size, size), 
                                dtype="double")
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O1(trans_perms)
    C = coo_array(
        (
            np.ones(N, dtype=int),
            (np.arange(N, dtype=int), atomic_decompr_idx),
        ),
        shape=(N, N // n_lp),
    )
    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        mat = spg_reps.get_sigma1_rep(i)
        mat = mat @ C
        mat = C.T @ mat
        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)
    return coset_reps_sum


