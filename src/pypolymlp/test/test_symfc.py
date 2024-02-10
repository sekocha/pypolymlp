#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import (
        phonopy_supercell,
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)

from pypolymlp.calculator.compute_fcs import recover_fc2, recover_fc3
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3


from scipy.sparse import csr_array, kron, coo_array

from symfc.spg_reps import SpgRepsO2
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.utils_O2 import (
    get_compr_coset_reps_sum,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_perm_compr_matrix,
    _get_atomic_lat_trans_decompr_indices,
)
import scipy

def compute_fc_basis_stable(supercell):

    ''' Constructing fc2 basis and fc3 basis '''
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set

    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    #fc3_basis = FCBasisSetO3(supercell, use_mkl=True).run()
    compress_mat_fc3 = fc3_basis.compression_matrix
    compress_eigvecs_fc3 = fc3_basis.basis_set
    t2 = time.time()
    print(' elapsed time (basis sets for fc2 and fc3) =', t2-t1)

#def test_coset(spg_reps):
#    trans_perms = spg_reps.translation_permutations
#    n_lp, N = trans_perms.shape
#    #size = N**2 * 9 // n_lp
#    size = N**2 * 9 
#    coset_reps_sum = csr_array(([], ([], [])), shape=(size, size), dtype="double")
#    atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)
#    C = csr_array(
#        (
#            np.ones(N**2, dtype=int),
#            (np.arange(N**2, dtype=int), atomic_decompr_idx),
#        ),
#        shape=(N**2, N**2 // n_lp),
#    )
#    #factor = 1 / n_lp / len(spg_reps.unique_rotation_indices)
#    factor = 1 / len(spg_reps.unique_rotation_indices)
#    print(len(spg_reps.unique_rotation_indices))
#    for i, _ in enumerate(spg_reps.unique_rotation_indices):
#        mat = spg_reps.get_sigma2_rep(i)
#        #mat = mat @ C
#        #mat = C.T @ mat
#        coset_reps_sum += kron(mat, spg_reps.r_reps[i] * factor)
#
#    return coset_reps_sum
    

def compute_fc_basis(supercell):
    
    spg_reps = SpgRepsO2(supercell)

    ''' get_c_trans() '''
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)

    c_perm = get_perm_compr_matrix(N)
    #coset_reps_sum = test_coset(spg_reps)
    c_pt = c_perm.T @ c_trans
    proj_pt = c_pt.T @ c_pt
    c_pt = eigsh_projector(proj_pt)
    print(c_pt.shape)

    coset_reps_sum = get_compr_coset_reps_sum(spg_reps)
    proj_rpt = c_pt.T @ coset_reps_sum @ c_pt
    c_rpt = eigsh_projector(proj_rpt)
    print(c_rpt.shape)

    proj_rpt2 = coset_reps_sum
    c_rpt2 = eigsh_projector(proj_rpt2)
    print(c_rpt2.shape)

    

#    p_trans = c_trans @ c_trans.T
#    p_perm = c_perm @ c_perm.T
#
#    p_zero = np.eye(N*N*9)
#
#    '''getrow and getcol are reversed in csr_array?'''
#    submat = coo_array(c_trans.getcol(10))
#    trans_ids = submat.row
#    for i in trans_ids:
#        p_zero[i,i] = 0.0
#
#    prod = p_trans @ p_zero @ p_trans
#    for i in range(10):
#        rank = int(round(sum(prod.diagonal())))
#        eigvals, eigvecs = scipy.sparse.linalg.eigsh(prod, k=rank+1)
#        print(eigvals)
#        prod = prod @ p_zero @ p_trans
#



#    p_zero = c_pt.T @ c_trans.T @ p_zero @ c_trans @ c_pt
#
#    print('perm, trans')
#    print(p_zero.shape, proj_pt.shape)
#    print(csr_array(p_zero @ proj_pt - proj_pt @ p_zero))
#
#    print('trans')
#    print(csr_array(p_zero @ p_trans - p_trans @ p_zero))
#
#    ##rank = int(round(sum(coset_reps_sum.diagonal())))
#    #eigvals, eigvecs = scipy.sparse.linalg.eigsh(coset_reps_sum, k=rank+1)
#    #print(eigvals)


def compute_fc_basis_from_structure(unitcell_dict=None, 
                                    supercell_dict=None, 
                                    supercell_matrix=None):

    if supercell_dict is not None:
        supercell = st_dict_to_phonopy_cell(supercell_dict)
    elif unitcell_dict is not None:
        supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
        supercell_dict = phonopy_cell_to_st_dict(supercell)

    compute_fc_basis(supercell)


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
    compute_fc_basis_from_structure(
            unitcell_dict=unitcell_dict,
            supercell_matrix=supercell_matrix
    )


