#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time
import gc
import math

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

'''
from pypolymlp.calculator.compute_fcs import recover_fc2, recover_fc3
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3
'''

from symfc.basis_sets.basis_sets_O3 import print_sp_matrix_size
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO3
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)

from symfc.utils.utils_O3 import (
    get_compr_coset_reps_sum_O3,
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)
from symfc.utils.matrix_tools_O3 import (
    get_perm_compr_matrix_O3,
)
import scipy
import time
#from sum_rules import compressed_projector_sum_rules


def compressed_complement_projector_sum_rules_lat_trans(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    '''Memory efficient algorithm'''

    n_lp, N = trans_perms.shape
    NNN27 = N**3 * 27
    NN27 = N**2 * 27

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((N,NN27)).T.reshape(-1)
    row = np.repeat(np.arange(NN27), N)
    c_sum_cplmt = csr_array(
        (
            np.ones(NNN27, dtype="double"), (row, decompr_idx),
        ),
        shape=(NN27, NNN27 // n_lp),
        dtype="double",
    )
    c_sum_cplmt = dot_product_sparse(
        c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl
    )
    proj_sum_cplmt = dot_product_sparse(
        c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
    )
    proj_sum_cplmt /= (n_lp * N)

    return proj_sum_cplmt

def compressed_complement_projector_sum_rules(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    """Return complementary projection matrix for sum rule compressed by C."""
    return compressed_complement_projector_sum_rules_lat_trans(
        n_a_compress_mat, trans_perms, use_mkl=use_mkl
    )


def compressed_projector_sum_rules(
    n_a_compress_mat: csr_array, trans_perms,  use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = compressed_complement_projector_sum_rules(
        n_a_compress_mat, trans_perms, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    t1 = time.time()
    t00 = time.time()

    '''space group representations'''
    spg_reps = SpgRepsO3(supercell)
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    print_sp_matrix_size(trans_perms, " trans_perms:")
    t01 = time.time()

    '''lattice translation'''
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    print_sp_matrix_size(c_trans, " C_(trans):")
    t02 = time.time()

    '''permutation'''
    c_perm = get_perm_compr_matrix_O3(N)
    print_sp_matrix_size(c_perm, " C_(perm):")
    t03 = time.time()

    c_pt = c_perm.T @ c_trans
    del c_perm
    del c_trans
    gc.collect()

    proj_pt = c_pt.T @ c_pt
    c_pt = eigsh_projector(proj_pt)
    del proj_pt
    gc.collect()

    print_sp_matrix_size(c_pt, " C_(perm,trans):")
    t04 = time.time()

    coset_reps_sum = get_compr_coset_reps_sum_O3(spg_reps)
    print_sp_matrix_size(coset_reps_sum, " R_(coset):")
    t05 = time.time()

    proj_rpt = c_pt.T @ coset_reps_sum @ c_pt
    del coset_reps_sum
    gc.collect()

    c_rpt = eigsh_projector(proj_rpt)
    del proj_rpt
    gc.collect()

    print_sp_matrix_size(c_rpt, " C_(perm,trans,coset):")
    t06 = time.time()

    n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)
    print_sp_matrix_size(n_a_compress_mat, " C_(n_a_compr):")

    t07 = time.time()

    proj = compressed_projector_sum_rules(n_a_compress_mat, 
                                          trans_perms, 
                                          use_mkl=True)
    print_sp_matrix_size(proj, " P_(perm,trans,coset,sum):")
    t08 = time.time()

    eigvecs = eigsh_projector_sumrule(proj)

    t09 = time.time()
    print("Basis size =", eigvecs.shape)
    print('-----')
    print('Time (spg. rep.)                        =', t01-t00)
    print('Time (lattice trans.)                   =', t02-t01)
    print('Time (permutation)                      =', t03-t02)
    print('Time (eigh(perm @ ltrans))              =', t04-t03)
    print('Time (coset)                            =', t05-t04)
    print('Time (eigh(coset @ perm @ ltrans))      =', t06-t05)
    print('Time (c_trans @ c_pt @ c_rpt)           =', t07-t06)
    print('Time (proj(coset @ perm @ ltrans @ sum) =', t08-t07)
    print('Time (eigh(coset @ perm @ ltrans @ sum) =', t09-t08)

    t2 = time.time()
    print('Elapsed time (basis sets for fc2 and fc3) =', t2-t1)


