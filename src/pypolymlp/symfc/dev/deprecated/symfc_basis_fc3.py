from symfc.utils.cutoff_tools import apply_zeros
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices


def permutation_dot_lat_trans_stable(trans_perms, fc_cutoff=None):
    """Simple implementation of permutation @ lattice translation"""
    if fc_cutoff is not None:
        zero_ids = fc_cutoff.find_zero_indices()
    else:
        zero_ids = None

    n_lp, N = trans_perms.shape
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)

    if zero_ids is not None:
        c_trans = apply_zeros(c_trans, zero_ids)
    print_sp_matrix_size(c_trans, " C_(trans):")

    c_perm = get_perm_compr_matrix_O3(N)
    print_sp_matrix_size(c_perm, " C_(perm):")

    c_pt = c_perm.T @ c_trans
    return c_pt


def run_basis_fc3(supercell, fc_cutoff=None, reduce_memory=True, apply_sum_rule=True):

    t00 = time.time()
    """space group representations"""
    print("Preparing SpgReps")
    spg_reps = SpgRepsO3Dev(supercell)
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    t01 = time.time()

    print("Preparing lattice translation")
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)
    print_sp_matrix_size(atomic_decompr_idx, " atomic_decompr_idx:")

    """permutation @ lattice translation"""
    if reduce_memory:
        proj_pt = projector_permutation_lat_trans_O3(
            trans_perms,
            atomic_decompr_idx=atomic_decompr_idx,
            fc_cutoff=fc_cutoff,
            use_mkl=True,
        )
    else:
        c_pt = permutation_dot_lat_trans_stable(trans_perms, fc_cutoff=fc_cutoff)
        print_sp_matrix_size(c_pt, " C_perm.T @ C_trans:")
        proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=True)

    print_sp_matrix_size(proj_pt, "P_(perm,trans):")
    t02 = time.time()

    c_pt = eigsh_projector(proj_pt)

    print_sp_matrix_size(c_pt, "C_(perm,trans):")
    t03 = time.time()

    proj_rpt = get_compr_coset_projector_O3(
        spg_reps,
        fc_cutoff=fc_cutoff,
        atomic_decompr_idx=atomic_decompr_idx,
        c_pt=c_pt,
    )
    t04 = time.time()

    c_rpt = eigsh_projector(proj_rpt)
    del proj_rpt
    gc.collect()

    print_sp_matrix_size(c_rpt, "C_(perm,trans,coset):")
    t05 = time.time()

    n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)
    print_sp_matrix_size(n_a_compress_mat, "C_(n_a_compr):")

    t06 = time.time()

    if apply_sum_rule:
        proj = compressed_projector_sum_rules_O3(
            trans_perms,
            n_a_compress_mat,
            atomic_decompr_idx=atomic_decompr_idx,
            fc_cutoff=fc_cutoff,
            use_mkl=True,
        )
        # proj = compressed_projector_sum_rules(
        #     trans_perms,
        #     n_a_compress_mat,
        #     use_mkl=True,
        #     atomic_decompr_idx=atomic_decompr_idx,
        #     fc_cutoff=fc_cutoff,
        # )
        """
        proj = compressed_projector_sum_rules_from_compact_compr_mat(
            trans_perms,
            n_a_compress_mat,
            use_mkl=True,
        )
        """
        print_sp_matrix_size(proj, "P_(perm,trans,coset,sum):")
        t07 = time.time()

        eigvecs = eigsh_projector_sumrule(proj)
        t08 = time.time()

    print("-----")
    print("Time (spg. rep.)                        =", "{:.3f}".format(t01 - t00))
    print("Time (proj(perm @ lattice trans.)       =", "{:.3f}".format(t02 - t01))
    print("Time (eigh(perm @ ltrans))              =", "{:.3f}".format(t03 - t02))
    print("Time (coset)                            =", "{:.3f}".format(t04 - t03))
    print("Time (eigh(coset @ perm @ ltrans))      =", "{:.3f}".format(t05 - t04))
    print("Time (c_pt @ c_rpt)                     =", "{:.3f}".format(t06 - t05))

    if apply_sum_rule:
        print("Time (proj(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t07 - t06))
        print("Time (eigh(coset @ perm @ ltrans @ sum) =", "{:.3f}".format(t08 - t07))
        print("Basis size =", eigvecs.shape)
        return n_a_compress_mat, eigvecs, atomic_decompr_idx

    print("Basis size =", n_a_compress_mat.shape)
    return n_a_compress_mat, proj_pt, atomic_decompr_idx


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
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    t1 = time.time()
    n_a_compress_mat, eigvecs = run_basis_fc3(supercell)
    # n_a_compress_mat = run_basis(supercell, apply_sum_rule=False)
    t2 = time.time()
    print("Elapsed time (basis sets for fc2 and fc3) =", t2 - t1)
