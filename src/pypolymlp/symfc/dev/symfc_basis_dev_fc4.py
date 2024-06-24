#!/usr/bin/env python
import argparse

# import gc
import signal
import time

import numpy as np
from symfc.utils.cutoff_tools import FCCutoff

from pypolymlp.symfc.dev.spg_reps_O4 import SpgRepsO4
from pypolymlp.symfc.dev.utils_O4 import get_atomic_lat_trans_decompr_indices_O4

# from symfc.utils.eig_tools import (
#     dot_product_sparse,
#     eigsh_projector,
#     eigsh_projector_sumrule,
# )


# from symfc.utils.matrix_tools_O3 import (
#     compressed_projector_sum_rules_O3,
#     get_perm_compr_matrix_O3,
#     projector_permutation_lat_trans_O3,
# )
# from symfc.utils.utils_O3 import (
#     get_atomic_lat_trans_decompr_indices_O3,
#     get_compr_coset_reps_sum_O3_dev,
#     get_lat_trans_compr_matrix_O3,
# )


# from pypolymlp.symfc.dev.matrix_tools_O2 import (
#     compressed_projector_sum_rules_O2,
#     projector_permutation_lat_trans_O2,
# )
# from pypolymlp.symfc.dev.utils_O2 import get_compr_coset_reps_sum_O2_dev


def run_basis_fc4(supercell, fc_cutoff=None):

    t00 = time.time()
    spg_reps = SpgRepsO4(supercell)
    trans_perms = spg_reps.translation_permutations
    t01 = time.time()

    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)
    print(len(atomic_decompr_idx))

    #    proj_pt = projector_permutation_lat_trans_O4(
    #        trans_perms,
    #        atomic_decompr_idx=atomic_decompr_idx,
    #        fc_cutoff=fc_cutoff,
    #        use_mkl=True,
    #    )
    t02 = time.time()
    #    c_pt = eigsh_projector(proj_pt)
    #
    #    proj_rpt = get_compr_coset_reps_sum_O2_dev(
    #        spg_reps,
    #        fc_cutoff=fc_cutoff,
    #        atomic_decompr_idx=atomic_decompr_idx,
    #        c_pt=c_pt,
    #    )
    #    c_rpt = eigsh_projector(proj_rpt)
    #    n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)
    #
    #    proj = compressed_projector_sum_rules_O2(
    #        trans_perms,
    #        n_a_compress_mat,
    #        atomic_decompr_idx=atomic_decompr_idx,
    #        fc_cutoff=fc_cutoff,
    #        use_mkl=True,
    #    )
    #    eigvecs = eigsh_projector_sumrule(proj)

    print("-----")
    print("Time (spg. rep.)                        =", "{:.3f}".format(t01 - t00))
    print("Time (proj(perm @ lattice trans.)       =", "{:.3f}".format(t02 - t01))
    #    print("Time (eigh(perm @ ltrans))              =", "{:.3f}".format(t03 - t02))
    #    print("Time (coset)                            =", "{:.3f}".format(t04 - t03))
    #    print("Time (eigh(coset @ perm @ ltrans))      =", "{:.3f}".format(t05 - t04))
    #    print("Time (c_pt @ c_rpt)                     =", "{:.3f}".format(t06 - t05))
    #
    #    if apply_sum_rule:
    #        print("Time (proj(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t07 - t06))
    #        print("Time (eigh(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t08 - t07))
    #        print("Basis size =", eigvecs.shape)
    #        return n_a_compress_mat, eigvecs, atomic_decompr_idx
    #
    #    print("Basis size =", n_a_compress_mat.shape)
    return None, None, None
    # return n_a_compress_mat, proj_pt, atomic_decompr_idx


#  def run_basis_fc3(supercell, fc_cutoff=None, reduce_memory=True, apply_sum_rule=True):
#
#      t00 = time.time()
#      """space group representations"""
#      print("Preparing SpgReps")
#      spg_reps = SpgRepsO3Dev(supercell)
#      trans_perms = spg_reps.translation_permutations
#      n_lp, N = trans_perms.shape
#      t01 = time.time()
#
#      print("Preparing lattice translation")
#      atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
#      print_sp_matrix_size(atomic_decompr_idx, " atomic_decompr_idx:")
#
#      """permutation @ lattice translation"""
#      if reduce_memory:
#          proj_pt = projector_permutation_lat_trans_O3(
#              trans_perms,
#              atomic_decompr_idx=atomic_decompr_idx,
#              fc_cutoff=fc_cutoff,
#              use_mkl=True,
#          )
#      else:
#          c_pt = permutation_dot_lat_trans_stable(trans_perms, fc_cutoff=fc_cutoff)
#          print_sp_matrix_size(c_pt, " C_perm.T @ C_trans:")
#          proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=True)
#
#      print_sp_matrix_size(proj_pt, "P_(perm,trans):")
#      t02 = time.time()
#
#      c_pt = eigsh_projector(proj_pt)
#
#      print_sp_matrix_size(c_pt, "C_(perm,trans):")
#      t03 = time.time()
#
#      proj_rpt = get_compr_coset_reps_sum_O3_dev(
#          spg_reps,
#          fc_cutoff=fc_cutoff,
#          atomic_decompr_idx=atomic_decompr_idx,
#          c_pt=c_pt,
#      )
#      t04 = time.time()
#
#      c_rpt = eigsh_projector(proj_rpt)
#      del proj_rpt
#      gc.collect()
#
#      print_sp_matrix_size(c_rpt, "C_(perm,trans,coset):")
#      t05 = time.time()
#
#      n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)
#      print_sp_matrix_size(n_a_compress_mat, "C_(n_a_compr):")
#
#      t06 = time.time()
#
#      if apply_sum_rule:
#          proj = compressed_projector_sum_rules_O3(
#              trans_perms,
#              n_a_compress_mat,
#              atomic_decompr_idx=atomic_decompr_idx,
#              fc_cutoff=fc_cutoff,
#              use_mkl=True,
#          )
#          # proj = compressed_projector_sum_rules(
#          #     trans_perms,
#          #     n_a_compress_mat,
#          #     use_mkl=True,
#          #     atomic_decompr_idx=atomic_decompr_idx,
#          #     fc_cutoff=fc_cutoff,
#          # )
#          """
#          proj = compressed_projector_sum_rules_from_compact_compr_mat(
#              trans_perms,
#              n_a_compress_mat,
#              use_mkl=True,
#          )
#          """
#          print_sp_matrix_size(proj, "P_(perm,trans,coset,sum):")
#          t07 = time.time()
#
#          eigvecs = eigsh_projector_sumrule(proj)
#          t08 = time.time()
#
#      print("-----")
#      print("Time (spg. rep.)                        =", "{:.3f}".format(t01 - t00))
#      print("Time (proj(perm @ lattice trans.)       =", "{:.3f}".format(t02 - t01))
#      print("Time (eigh(perm @ ltrans))              =", "{:.3f}".format(t03 - t02))
#      print("Time (coset)                            =", "{:.3f}".format(t04 - t03))
#      print("Time (eigh(coset @ perm @ ltrans))      =", "{:.3f}".format(t05 - t04))
#      print("Time (c_pt @ c_rpt)                     =", "{:.3f}".format(t06 - t05))
#
#      if apply_sum_rule:
#          print("Time (proj(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t07 - t06))
#          print("Time (eigh(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t08 - t07))
#          print("Basis size =", eigvecs.shape)
#          return n_a_compress_mat, eigvecs, atomic_decompr_idx
#
#      print("Basis size =", n_a_compress_mat.shape)
#      return n_a_compress_mat, proj_pt, atomic_decompr_idx


if __name__ == "__main__":

    from pypolymlp.core.interface_vasp import Poscar
    from pypolymlp.utils.phonopy_utils import phonopy_supercell

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff radius for setting zero elements.",
    )
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    if args.cutoff is not None:
        fc_cutoff = FCCutoff(supercell, cutoff=args.cutoff)
    else:
        fc_cutoff = None

    t1 = time.time()
    n_a_compress_mat, eigvecs = run_basis_fc4(supercell, fc_cutoff=fc_cutoff)
    t2 = time.time()
    print("Elapsed time (basis sets for fc4) =", t2 - t1)
