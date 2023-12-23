#!/usr/bin/env python
import numpy as np
import argparse
import time

from pypolymlp.symfc_dev.linalg import solve_eig
from pypolymlp.symfc_dev.linalg import dot_sp_mats
from pypolymlp.symfc_dev.cell import poscar_to_supercell
from pypolymlp.symfc_dev.cell import st_dict_to_phonony
from pypolymlp.symfc_dev.basis_set_O3 import run_fc3

def fc3_projector_kp_symfc(spg_reps3):
    coset_reps_sum = get_compr_coset_reps_sum_O3(spg_reps3)
    trans_perms = spg_reps3.translation_permutations
    n_lp, N = trans_perms.shape
    size = 27 * N**3

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    c_trans = get_lat_trans_compr_matrix_O3(decompr_idx, N, n_lp)
    proj = dot_sp_mats(coset_reps_sum, c_trans.tocsr().transpose())
    proj = dot_sp_mats(c_trans.tocsr(), proj)
    return proj

def fc3_projector_perm(N):
    c_perm = permutation_symmetry_basis(N)
    return dot_sp_mats(c_perm, c_perm.transpose())

def test_fc3_projector_kp_perm_symfc(spg_reps3):

    trans_perms = spg_reps3.translation_permutations
    n_lp, N = trans_perms.shape

    proj_kp = fc3_projector_kp_symfc(spg_reps3)
    proj_perm = fc3_projector_perm(N)

    proj1 = dot_sp_mats(proj_kp, proj_perm)
    proj2 = dot_sp_mats(proj_perm, proj_kp)

    comm = proj1 - proj2
    print(' comm (nonzero) =', np.where(np.abs(comm.data) > 1e-15)[0])

    print(' eigs: P_KP (symfc)')
    eigvals, eigvecs = solve_eig(proj_kp)

    print(' eigs: P_KP (symfc) @ P_perm')
    eigvals, eigvecs = solve_eig(proj1)

    return eigvals, eigvecs, proj_kp

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
    parser.add_argument('--calc_proj',
                        action='store_true',
                        help='Full projector calculations')
    parser.add_argument('--mkl',
                        action='store_true',
                        help='use mkl')
    args = parser.parse_args()

    supercell_mat = np.array(args.supercell).reshape([3,3])
    unitcell, supercell = poscar_to_supercell(args.poscar, supercell_mat)
    supercell_phonopy = st_dict_to_phonony(supercell)

    if args.calc_proj:
        eigvals, eigvecs, proj_kp = test_fc3_projector_kp_perm_symfc(spg_reps3)

    t1 = time.time()
    compress_mat, eigvecs = run_fc3(supercell_phonopy, mkl=args.mkl)
    t2 = time.time()
    print(' elapsed time (rot + trans + perm + sum) :', t2-t1, '(s)')


