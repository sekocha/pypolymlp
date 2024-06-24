"""Utility functions for 3rd order force constants."""

import numpy as np
from scipy.sparse import csr_array, kron
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices

from pypolymlp.symfc.dev.spg_reps_O2_dev import SpgRepsO2Dev


def get_compr_coset_reps_sum_O2_dev(
    spg_reps: SpgRepsO2Dev,
    fc_cutoff: FCCutoff = None,
    atomic_decompr_idx: np.ndarray = None,
    c_pt: csr_array = None,
) -> csr_array:
    """Return compr matrix of sum of coset reps."""
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    size = N**2 * 9 // n_lp if c_pt is None else c_pt.shape[1]
    coset_reps_sum = csr_array((size, size), dtype="double")

    if atomic_decompr_idx is None:
        print("Preparing lattice_translation")
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    if fc_cutoff is None:
        nonzero = None
        size_data = N**2
    else:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc2()
        size_data = np.count_nonzero(nonzero)

    factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
    for i, _ in enumerate(spg_reps.unique_rotation_indices):
        permutation = spg_reps.get_sigma2_rep(i, nonzero=nonzero)
        if nonzero is None:
            mat = csr_array(
                (
                    np.ones(size_data, dtype="int_"),
                    (atomic_decompr_idx[permutation], atomic_decompr_idx),
                ),
                shape=(N**2 // n_lp, N**2 // n_lp),
                dtype="int_",
            )
        else:
            mat = csr_array(
                (
                    np.ones(size_data, dtype="int_"),
                    (atomic_decompr_idx[permutation], atomic_decompr_idx[nonzero]),
                ),
                shape=(N**2 // n_lp, N**2 // n_lp),
                dtype="int_",
            )

        mat = kron(mat, spg_reps.r_reps[i] * factor)
        if c_pt is not None:
            mat = c_pt.T @ mat @ c_pt

        coset_reps_sum += mat

    return coset_reps_sum
