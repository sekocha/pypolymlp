#!/usr/bin/env python
import numpy as np
import time

from pypolymlp.symfc_dev.spg_reps_O3 import SpgRepsO3
from pypolymlp.symfc_dev.utils_O3 import get_compr_coset_reps_sum_O3
from pypolymlp.symfc_dev.utils_O3 import get_lat_trans_compr_matrix_O3
from pypolymlp.symfc_dev.utils_O3 import get_lat_trans_decompr_indices_O3

from pypolymlp.symfc_dev.matrix_O3 import permutation_symmetry_basis
from pypolymlp.symfc_dev.matrix_O3 import compressed_projector_sum_rules
from pypolymlp.symfc_dev.linalg import eigsh_projector
from pypolymlp.symfc_dev.linalg import eigsh_projector_sumrule
from pypolymlp.symfc_dev.linalg import dot_sp_mats

def print_sp_matrix_size(c, header):
    print(header, c.shape, len(c.data))

def run_fc3(supercell_phonopy, mkl=False, return_trans=False):

    spg_reps3 = SpgRepsO3(supercell_phonopy)
    trans_perms = spg_reps3.translation_permutations
    n_lp, N = trans_perms.shape

    tt00 = time.time()
    """C(permutation)"""
    c_perm = permutation_symmetry_basis(N)
    print_sp_matrix_size(c_perm, ' C_perm:')
    tt0 = time.time()

    """C(trans)"""
    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    c_trans = get_lat_trans_compr_matrix_O3(decompr_idx, N, n_lp)
    print_sp_matrix_size(c_trans, ' C_trans:')
    tt1 = time.time()

    """C(pt) = C(perm).T @ C(trans)"""
    c_pt = dot_sp_mats(c_perm.transpose(), c_trans)
    print_sp_matrix_size(c_pt, ' C_(perm,trans):')
    proj_pt = dot_sp_mats(c_pt.transpose(), c_pt)
    print_sp_matrix_size(proj_pt, ' P_(perm,trans):')
    tt2 = time.time()

    coset_reps_sum = get_compr_coset_reps_sum_O3(spg_reps3)
    print_sp_matrix_size(coset_reps_sum, ' R_(coset):')
    tt3 = time.time()

    '''
    compression using C(pt) 
        = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
    proj_rpt = c_pt.transpose() @ coset_reps_sum @ c_pt
    '''
    c_pt = eigsh_projector(proj_pt)
    print_sp_matrix_size(c_pt, ' C_(perm,trans,compressed):')

    proj_rpt = dot_sp_mats(coset_reps_sum, c_pt)
    proj_rpt = dot_sp_mats(c_pt.transpose(), proj_rpt)
    print_sp_matrix_size(proj_rpt, ' P_(perm,trans,coset):')
    tt4 = time.time()

    c_rpt = eigsh_projector(proj_rpt)
    print_sp_matrix_size(c_rpt, ' C_(perm,trans,coset):')
    tt5 = time.time()

    '''
    [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
     = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
     = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
     = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
     = C_rpt @ proj
        compress_mat = c_trans @ c_pt @ c_rpt
    '''
    compress_mat_wo_trans = dot_sp_mats(c_pt, c_rpt)
    compress_mat = dot_sp_mats(c_trans, compress_mat_wo_trans)
    #compress_mat = dot_sp_mats(c_trans, c_pt)
    #compress_mat = dot_sp_mats(compress_mat, c_rpt)
    print_sp_matrix_size(compress_mat, ' compression matrix:')

    proj = compressed_projector_sum_rules(compress_mat, N, mkl=mkl)
    print_sp_matrix_size(proj, ' P_(perm,trans,coset,sum):')
    tt6 = time.time()

    eigvecs = eigsh_projector_sumrule(proj)
    tt7 = time.time()
    print(' basis (size) =', eigvecs.shape)

    print('  t (init., perm)         = ', tt0-tt00)
    print('  t (init., trans)        = ', tt1-tt0)
    print('  t (dot, trans, perm)    = ', tt2-tt1)
    print('  t (coset_reps_sum)      = ', tt3-tt2)
    print('  t (dot, coset_reps_sum) = ', tt4-tt3)
    print('  t (rot, trans, perm)    = ', tt5-tt4)
    print('  t (proj_st)             = ', tt6-tt5)
    print('  t (eigh(svd))           = ', tt7-tt6)
    if return_trans:
        return c_trans, compress_mat_wo_trans, compress_mat, eigvecs
    return compress_mat, eigvecs

def recover_fc3(coefs, compress_mat, compress_eigvecs, N):
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((N,N,N,3,3,3))
    return fc3

