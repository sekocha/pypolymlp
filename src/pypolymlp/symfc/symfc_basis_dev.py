#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time
import gc
import math
import itertools

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
    N3N3N3_to_NNN333,
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


def permutation_dot_lat_trans_stable(trans_perms):

    n_lp, N = trans_perms.shape
    c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
    print_sp_matrix_size(c_trans, " C_(trans):")

    c_perm = get_perm_compr_matrix_O3(N)
    print_sp_matrix_size(c_perm, " C_(perm):")

    c_pt = c_perm.T @ c_trans
    del c_perm
    del c_trans
    gc.collect()
    return c_pt


def permutation_dot_lat_trans(trans_perms):

    n_lp, natom = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    NNN27 = natom**3 * 27

    combinations3 = np.array(
        list(itertools.combinations(range(3 * natom), 3)), dtype=int
    )
    combinations2 = np.array(
        list(itertools.combinations(range(3 * natom), 2)), dtype=int
    )
    combinations1 = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)

    n_perm3 = combinations3.shape[0]
    n_perm2 = combinations2.shape[0] * 2
    n_perm1 = combinations1.shape[0]
    n_perm = n_perm3 + n_perm2 + n_perm1

    n_data3 = combinations3.shape[0] * 6
    n_data2 = combinations2.shape[0] * 6
    n_data1 = combinations1.shape[0]
    n_data = n_data3 + n_data2 + n_data1

    row = np.zeros(n_data, dtype="int_")
    col = np.zeros(n_data, dtype="int_")
    data = np.zeros(n_data, dtype="double")

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    begin_id, end_id = 0, n_data3
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    combinations_perm = combinations3[:, perms].reshape((-1, 3))
    combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

    row[begin_id:end_id] = np.repeat(range(n_perm3), 6)
    col[begin_id:end_id] = decompr_idx[combinations_perm]
    data[begin_id:end_id] = 1 / math.sqrt(6 * n_lp)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    begin_id = end_id
    end_id = begin_id + n_data2
    perms = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    combinations_perm = combinations2[:, perms].reshape((-1, 3))
    combinations_perm = N3N3N3_to_NNN333(combinations_perm, natom)

    row[begin_id:end_id] = np.repeat(range(n_perm3, n_perm3 + n_perm2), 3)
    col[begin_id:end_id] = decompr_idx[combinations_perm]
    data[begin_id:end_id] = 1 / math.sqrt(3 * n_lp)

    # (1) for FC3 with single index ia
    begin_id = end_id
    combinations_perm = N3N3N3_to_NNN333(combinations1, natom)
    row[begin_id:] = np.array(range(n_perm3 + n_perm2, n_perm))
    col[begin_id:] = decompr_idx[combinations_perm]
    data[begin_id:] = 1.0 / math.sqrt(n_lp)

    c_pt = csr_array(
        (data, (row, col)),
        shape=(n_perm, NNN27 // n_lp),
        dtype="double",
    )
    return c_pt


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

    '''permutation @ lattice translation'''
    #c_pt = permutation_dot_lat_trans_stable(trans_perms)
    c_pt = permutation_dot_lat_trans(trans_perms)
    print_sp_matrix_size(c_pt, " C_perm.T @ C_trans:")
    t02 = time.time()

    #proj_pt = c_pt.T @ c_pt
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=True)
    print_sp_matrix_size(proj_pt, " P_(perm,trans):")
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
    print('Time (perm @ lattice trans.)            =', t02-t01)
    print('Time (eigh(perm @ ltrans))              =', t04-t02)
    print('Time (coset)                            =', t05-t04)
    print('Time (eigh(coset @ perm @ ltrans))      =', t06-t05)
    print('Time (c_pt @ c_rpt)                     =', t07-t06)
    print('Time (proj(coset @ perm @ ltrans @ sum) =', t08-t07)
    print('Time (eigh(coset @ perm @ ltrans @ sum) =', t09-t08)

    t2 = time.time()
    print('Elapsed time (basis sets for fc2 and fc3) =', t2-t1)


