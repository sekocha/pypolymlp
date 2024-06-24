from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices

from pypolymlp.symfc.dev.matrix_tools_O2 import (
    compressed_projector_sum_rules_O2,
    projector_permutation_lat_trans_O2,
)
from pypolymlp.symfc.dev.spg_reps_O2_dev import SpgRepsO2Dev
from pypolymlp.symfc.dev.utils_O2 import get_compr_coset_reps_sum_O2_dev


def run_basis_fc2(supercell, fc_cutoff=None):

    spg_reps = SpgRepsO2Dev(supercell)
    trans_perms = spg_reps.translation_permutations
    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    proj_pt = projector_permutation_lat_trans_O2(
        trans_perms,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=fc_cutoff,
        use_mkl=True,
    )
    c_pt = eigsh_projector(proj_pt)

    proj_rpt = get_compr_coset_reps_sum_O2_dev(
        spg_reps,
        fc_cutoff=fc_cutoff,
        atomic_decompr_idx=atomic_decompr_idx,
        c_pt=c_pt,
    )
    c_rpt = eigsh_projector(proj_rpt)
    n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)

    proj = compressed_projector_sum_rules_O2(
        trans_perms,
        n_a_compress_mat,
        atomic_decompr_idx=atomic_decompr_idx,
        fc_cutoff=fc_cutoff,
        use_mkl=True,
    )
    eigvecs = eigsh_projector_sumrule(proj)
    return n_a_compress_mat, eigvecs, atomic_decompr_idx
