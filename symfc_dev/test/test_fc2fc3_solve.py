#!/usr/bin/env python
import numpy as np
import sys
import time

from pypolymlp.symfc_dev.cell import st_dict_to_phonony
from pypolymlp.symfc_dev.data import parse_dataset, parse_dataset_phono3py_xz
from pypolymlp.symfc_dev.basis_set_O2 import run_fc2, recover_fc2
from pypolymlp.symfc_dev.basis_set_O3 import run_fc3, recover_fc3
from pypolymlp.symfc_dev.solver_O2O3 import run_solver_sparse_O2O3

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
    disps = disps[:1000]
    forces = forces[:1000]
    print(disps.shape, forces.shape)

    ''' Constructing fc2 basis '''
    t1 = time.time()
    compress_mat_fc2, compress_eigvecs_fc2 = run_fc2(supercell_phonopy, 
                                                     mkl=False)
    t2 = time.time()
    print(' elapsed time (basis fc2) =', t2-t1)

    ''' Constructing fc3 basis '''
    t1 = time.time()
    compress_mat_fc3, compress_eigvecs_fc3  = run_fc3(supercell_phonopy, 
                                                      mkl=True)
    t2 = time.time()
    print(' elapsed time (basis fc3) =', t2-t1)

    ''' Solving fc3 using run_solver_sparse '''
    print('-----')
    t1 = time.time()
    coefs_fc2, coefs_fc3 = run_solver_sparse_O2O3(disps, 
                                                   forces, 
                                                   compress_mat_fc2, 
                                                   compress_mat_fc3, 
                                                   compress_eigvecs_fc2,
                                                   compress_eigvecs_fc3,
                                                   use_mkl=True,
                                                   batch_size=200)
    t2 = time.time()
    print(' elapsed time (solve fc2 + fc3) =', t2-t1)

    fc2 = recover_fc2(coefs_fc2, compress_mat_fc2, compress_eigvecs_fc2, N)
    fc3 = recover_fc3(coefs_fc3, compress_mat_fc3, compress_eigvecs_fc3, N)
    write_fc2_to_hdf5(fc2)
    write_fc3_to_hdf5(fc3)

