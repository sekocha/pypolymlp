#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

'''
from pypolymlp.calculator.compute_fcs import recover_fc2, recover_fc3
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3
'''

from symfc.basis_sets.basis_sets_O3 import print_sp_matrix_size
from scipy.sparse import csr_array, kron, coo_array

from symfc.spg_reps import SpgRepsO3
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)

from symfc.utils.utils_O3 import (
    get_compr_coset_reps_sum_O3,
    get_lat_trans_compr_matrix_O3,
)
from symfc.utils.matrix_tools_O3 import (
    compressed_projector_sum_rules,
    get_perm_compr_matrix_O3,
)
import scipy


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
    t01 = time.time()

    '''lattice translation'''
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    t02 = time.time()

    '''permutation'''
    c_perm = get_perm_compr_matrix_O3(N)
    t03 = time.time()

    c_pt = c_perm.T @ c_trans
    proj_pt = c_pt.T @ c_pt
    c_pt = eigsh_projector(proj_pt)
    print_sp_matrix_size(c_pt, " C_(perm,trans):")
    t04 = time.time()

    coset_reps_sum = get_compr_coset_reps_sum_O3(spg_reps)
    t05 = time.time()

    proj_rpt = c_pt.T @ coset_reps_sum @ c_pt
    c_rpt = eigsh_projector(proj_rpt)
    print_sp_matrix_size(c_rpt, " C_(perm,trans,coset):")
    t06 = time.time()

    n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=True)
    compress_mat = dot_product_sparse(
            c_trans, n_a_compress_mat, use_mkl=True,
    )
    t07 = time.time()

    proj = compressed_projector_sum_rules(compress_mat, N, use_mkl=True)
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


