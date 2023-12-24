#!/usr/bin/env python
import numpy as np
import argparse
import time

import scipy

from pypolymlp.symfc_dev.cell import poscar_to_supercell
from pypolymlp.symfc_dev.cell import st_dict_to_phonony

from symfc.spg_reps import SpgReps
from pypolymlp.symfc_dev.spg_reps_O4 import SpgRepsO4
from pypolymlp.symfc_dev.utils_O4 import get_compr_coset_reps_sum_O4
from pypolymlp.symfc_dev.utils_O4 import get_lat_trans_compr_matrix_O4
from pypolymlp.symfc_dev.utils_O4 import get_lat_trans_decompr_indices_O4

from pypolymlp.symfc_dev.matrix_O4 import permutation_symmetry_basis
from pypolymlp.symfc_dev.matrix_O4 import compressed_projector_sum_rules
from pypolymlp.symfc_dev.linalg import solve_eig
from pypolymlp.symfc_dev.linalg import eigsh_projector
from pypolymlp.symfc_dev.linalg import eigsh_projector_memory_efficient
from pypolymlp.symfc_dev.linalg import eigsh_projector_sumrule
from pypolymlp.symfc_dev.linalg import dot_sp_mats

def print_sp_matrix_size(c, header):
    print(header, c.shape, len(c.data))

def run_fc4(spg_reps4, compress_pt=True):

    trans_perms = spg_reps4.translation_permutations
    n_lp, N = trans_perms.shape

    tt00 = time.time()
    # C(permutation)
    c_perm = permutation_symmetry_basis(N)
    print_sp_matrix_size(c_perm, ' C_perm:')
    tt0 = time.time()

    # C(trans)
    decompr_idx = get_lat_trans_decompr_indices_O4(trans_perms)
    c_trans = get_lat_trans_compr_matrix_O4(decompr_idx, N, n_lp)
    print_sp_matrix_size(c_trans, ' C_trans:')
    tt1 = time.time()

    # C(pt) = C(perm).T @ C(trans)
    c_pt = dot_sp_mats(c_perm.transpose(), c_trans)
    #c_pt = dot_sp_mats(c_perm.transpose().tocsr(), c_trans.tocsr(), mkl=True)
    print_sp_matrix_size(c_pt, ' C_(perm,trans):')
    proj_pt = dot_sp_mats(c_pt.transpose(), c_pt)
    #proj_pt = dot_sp_mats(c_pt.transpose(), c_pt, mkl=True)
    print_sp_matrix_size(proj_pt, ' P_(perm,trans):')
    tt2 = time.time()

    coset_reps_sum = get_compr_coset_reps_sum_O4(spg_reps4)
    print_sp_matrix_size(coset_reps_sum, ' R_(coset_reps_sum):')
    tt3 = time.time()

    if compress_pt:
        '''
         compression using C(pt) 
           = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        '''
        c_pt = eigsh_projector_memory_efficient(proj_pt)
        print_sp_matrix_size(c_pt, ' C_(perm,trans,compressed):')

        '''
        proj = c_pt.transpose() @ coset_reps_sum @ c_pt
        '''
        proj = dot_sp_mats(coset_reps_sum, c_pt)
        proj = dot_sp_mats(c_pt.transpose(), proj)
    else:
        proj = dot_sp_mats(proj_pt, coset_reps_sum)
    print_sp_matrix_size(proj, ' P_(perm,trans,rot):')
    tt4 = time.time()

    #c_rpt = eigsh_projector_memory_efficient(proj)
    c_rpt = eigsh_projector(proj)
    print_sp_matrix_size(c_rpt, ' C_(perm,trans,rot):')
    tt5 = time.time()

    '''
    [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
     = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
     = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
     = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
     = C_rpt @ proj
    '''

    if compress_pt:
        '''
        compress_mat = c_trans @ c_pt @ c_rpt
        '''
        compress_mat = dot_sp_mats(c_trans, c_pt)
        compress_mat = dot_sp_mats(compress_mat, c_rpt)
    else:
        compress_mat = dot_sp_mats(c_trans, c_rpt)
    print_sp_matrix_size(compress_mat, ' compression matrix:')

    #proj = compressed_projector_sum_rules(compress_mat, N, mkl=False)
    proj = compressed_projector_sum_rules(compress_mat, N, mkl=True)
    print_sp_matrix_size(proj, ' P_(perm,trans,rot,sum):')
    tt6 = time.time()

    eigvecs = eigsh_projector_sumrule(proj)
    print(' basis (size) =', eigvecs.shape)

    tt7 = time.time()
    #full_eigvecs = dot_sp_mats(compress_mat, csr_matrix(eigvecs))
    tt8 = time.time()

    print('  t (init., perm)         = ', tt0-tt00)
    print('  t (init., trans)        = ', tt1-tt0)
    print('  t (dot, trans, perm)    = ', tt2-tt1)
    print('  t (coset_reps_sum)      = ', tt3-tt2)
    print('  t (dot, coset_reps_sum) = ', tt4-tt3)
    print('  t (rot, trans, perm)    = ', tt5-tt4)
    print('  t (proj_st)             = ', tt6-tt5)
    print('  t (eigh(svd))           = ', tt7-tt6)
#    #print('  t (reconstruction)      = ', tt8-tt7)

    eigvecs = c_rpt
    return eigvecs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        type=str,
                        default='POSCAR',
                        help='poscar file name')
    parser.add_argument('--supercell',
                        type=int,
                        nargs=9,
                        default=[2,0,0,0,2,0,0,0,2],
                        help='Supercell size')
    args = parser.parse_args()

    supercell_mat = np.array(args.supercell).reshape([3,3])
    unitcell, supercell = poscar_to_supercell(args.poscar, supercell_mat)
    supercell_phonopy = st_dict_to_phonony(supercell)
    spg_reps4 = SpgRepsO4(supercell_phonopy)

    t1 = time.time()
    eigvecs = run_fc4(spg_reps4)
    t2 = time.time()
    print(' rot + trans + perm + sum :', t2-t1, '(s)')


