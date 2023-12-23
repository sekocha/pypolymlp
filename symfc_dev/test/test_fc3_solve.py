#!/usr/bin/env python
import numpy as np
import sys
import time

from pypolymlp.symfc_dev.cell import st_dict_to_phonony
from pypolymlp.symfc_dev.data import parse_dataset, parse_dataset_phono3py_xz
from pypolymlp.symfc_dev.basis_set_O2 import run_fc2, recover_fc2
from pypolymlp.symfc_dev.basis_set_O3 import run_fc3, recover_fc3

from pypolymlp.symfc_dev.solver_O2 import run_solver_dense_O2
from pypolymlp.symfc_dev.solver_O3 import run_solver_dense_O3
from pypolymlp.symfc_dev.solver_O3 import run_solver_sparse_O3

from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    try:
        filename = sys.argv[1]
        disps, forces, supercell_phonopy = parse_dataset_phono3py_xz(filename)
    except:
        disps, forces, supercell = parse_dataset()
        supercell_phonopy = st_dict_to_phonony(supercell)
    N = len(supercell_phonopy.numbers)

    ''' Constructing fc2 basis and solving fc2 '''
    t1 = time.time()
    compress_mat, compress_eigvecs = run_fc2(supercell_phonopy, mkl=True)
    coefs_fc2 = run_solver_dense_O2(disps, 
                                    forces, 
                                    compress_mat, 
                                    compress_eigvecs)
    fc2 = recover_fc2(coefs_fc2, compress_mat, compress_eigvecs, N)
    write_fc2_to_hdf5(fc2)
    t2 = time.time()
    print(' elapsed time (basis fc2 + solve fc2) =', t2-t1)

    forces_update = []
    for f, u in zip(forces, disps):
        fc2_mat = fc2.transpose((0,2,1,3)).reshape(N*3,N*3)
        forces_update.append(f + np.dot(fc2_mat, u))
    forces = np.array(forces_update)
    print('-- fc2: finished --')

    ''' Constructing fc3 basis '''
    t1 = time.time()
    compress_mat, compress_eigvecs = run_fc3(supercell_phonopy, mkl=True)
    t2 = time.time()
    print(' elapsed time (compute compr. basis) =', t2-t1)

    ''' Solving fc3 using run_solver_sparse '''
    print('-----')
    t1 = time.time()
    coefs_sp = run_solver_sparse_O3(disps, 
                                    forces, 
                                    compress_mat, 
                                    compress_eigvecs,
                                    batch_size=200)
    t2 = time.time()
    print(' elapsed time (solve fc3, sparse) =', t2-t1)
    fc3_sp = recover_fc3(coefs_sp, compress_mat, compress_eigvecs, N)
    write_fc3_to_hdf5(fc3_sp)

    ''' Solving fc3 using run_solver_dense (slow algorithm) '''
    dense = False
    if compress_eigvecs.shape[1] < 300 or dense:
        print('-----')
        t3 = time.time()
        coefs_dense = run_solver_dense_O3(disps, forces, 
                                          compress_mat, compress_eigvecs)
        t4 = time.time()
        print(' elapsed time (solve fc3, dense) =', t4-t3)

        fc3_dense = recover_fc3(coefs_dense, compress_mat, compress_eigvecs, N)
        print(' allclose (fc3,sp&dense) =', np.allclose(coefs_dense, coefs_sp))


