#!/usr/bin/env python
import numpy as np
import time

from pypolymlp.symfc_dev.cell import st_dict_to_phonony
from pypolymlp.symfc_dev.data import parse_dataset
from pypolymlp.symfc_dev.basis_set_O2 import run_fc2, recover_fc2
from pypolymlp.symfc_dev.solver_O2 import run_solver_dense_O2, solve
from pypolymlp.symfc_dev.solver_O2 import run_solver_sparse_O2

if __name__ == '__main__':

    ''' algorithm 1'''
    full_basis1 = np.load('fc_basis.npy')
    disps, forces, supercell = parse_dataset()
    supercell_mat = supercell['supercell_matrix']
    N = len(supercell['elements'])
    print(full_basis1.shape)
    print(disps.shape, forces.shape)

    coefs_true1 = solve(disps, forces, full_basis1)
    np.set_printoptions(suppress=True)
    print(coefs_true1)
    fc1 = full_basis1.transpose((1,2,3,4,0)) @ coefs_true1

    ''' algorithm 2'''
    supercell_phonopy = st_dict_to_phonony(supercell)
    compress_mat, compress_eigvecs = run_fc2(supercell_phonopy, mkl=True)
    print(compress_mat.shape, compress_eigvecs.shape)
    t1 = time.time()
    coefs_true2 = run_solver_dense_O2(disps,
                                      forces,
                                      compress_mat,
                                      compress_eigvecs)
    t2 = time.time()
    print(' elapsed time (solve fc2, dense) =', t2-t1)
    print(coefs_true2)
    fc2 = recover_fc2(coefs_true2, compress_mat, compress_eigvecs, N)
    print(np.allclose(fc1, fc2))

    ''' algorithm 3'''
    t1 = time.time()
    coefs_true2 = run_solver_sparse_O2(disps,
                                       forces,
                                       compress_mat,
                                       compress_eigvecs)
    t2 = time.time()
    print(' elapsed time (solve fc2, sparse) =', t2-t1)
    print(coefs_true2)
    fc2 = recover_fc2(coefs_true2, compress_mat, compress_eigvecs, N)
    print(np.allclose(fc1, fc2))
 
