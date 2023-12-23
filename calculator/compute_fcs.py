#!/usr/bin/env python 
import numpy as np
import argparse
import time

from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.calculator.compute_properties import compute_properties

from lammps_api.sscha.branch.basis_set_O2 import run_fc2, recover_fc2
from lammps_api.sscha.branch.basis_set_O3 import run_fc3, recover_fc3
from lammps_api.sscha.branch.solver_O2O3 import run_solver_sparse_O2O3

from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5


def compute_fcs(pot, phono3py_yaml=None):

    if phono3py_yaml is not None:
        supercell_phonopy, disps, st_dicts = parse_phono3py_yaml_fcs(
                                                phono3py_yaml
                                            )
    else:
        pass
        '''
        #from lammps_api.sscha.branch.cell import st_dict_to_phonony
        random displacements + POSCAR --> supercell_phonopy, st_dicts
        '''

    t1 = time.time()
    _, forces, _ = compute_properties(pot, st_dicts)
    t2 = time.time()
    print(' elapsed time (computing forces) =', t2-t1)

    forces = np.array(forces)
    n_data, N, _ = forces.shape
    disps = disps.reshape((n_data, -1))
    forces = forces.reshape((n_data, -1))

    '''
    norms = np.array([np.linalg.norm(f1-f2)
                      for i, (f1, f2) in enumerate(zip(forces, forces_dft))])
    ids = np.where(norms > 0.01)[0]
    print(ids)
    print(norms[ids])
    '''

    ''' Constructing fc2 basis '''
    compress_mat_fc2, compress_eigvecs_fc2 = run_fc2(supercell_phonopy, 
                                                     mkl=False)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pot',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')
    parser.add_argument('--phono3py_yaml',
                        type=str,
                        default=None,
                        help='phono3py_yaml files')
    args = parser.parse_args()

    compute_fcs(args.pot, phono3py_yaml=args.phono3py_yaml)
