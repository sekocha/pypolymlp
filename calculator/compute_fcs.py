#!/usr/bin/env python 
import numpy as np
import argparse
import time

from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import (
        phonopy_supercell,
        phonopy_cell_to_st_dict,
)
from pypolymlp.utils.displacements_utils import generate_random_displacements

from pypolymlp.calculator.compute_properties import compute_properties
from pypolymlp.symfc_dev.basis_set_O2 import run_fc2, recover_fc2
from pypolymlp.symfc_dev.basis_set_O3 import run_fc3, recover_fc3
from pypolymlp.symfc_dev.solver_O2O3 import run_solver_sparse_O2O3

from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

def compute_fcs(pot, 
                phono3py_yaml=None, 
                st_dict=None, 
                supercell_matrix=None,
                n_samples=100,
                displacements=0.03):

    if phono3py_yaml is not None:
        supercell, disps, st_dicts = parse_phono3py_yaml_fcs(phono3py_yaml)
    elif st_dict is not None:
        supercell = phonopy_supercell(st_dict, supercell_matrix)
        disps, st_dicts = generate_random_displacements(
                phonopy_cell_to_st_dict(supercell),
                n_samples=n_samples,
                displacements=displacements
        )

    ''' disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
    disps = disps.transpose((0,2,1)) 

    t1 = time.time()
    ''' forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
    _, forces, _ = compute_properties(pot, st_dicts)
    forces = np.array(forces).transpose((0,2,1)) 
    t2 = time.time()
    print(' elapsed time (computing forces) =', t2-t1)

    n_data, N, _ = forces.shape
    disps = disps.reshape((n_data, -1))
    forces = forces.reshape((n_data, -1))

    ''' Constructing fc2 basis and fc3 basis '''
    compress_mat_fc2, compress_eigvecs_fc2 = run_fc2(supercell, mkl=False)
    t1 = time.time()
    compress_mat_fc3, compress_eigvecs_fc3  = run_fc3(supercell, mkl=True)
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
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help='Number of random displacement samples')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size')
    args = parser.parse_args()

    if args.poscar is not None:
        st_dict = Poscar(args.poscar).get_structure()
    else:
        st_dict = None

    supercell_matrix = np.diag(args.supercell)
    compute_fcs(args.pot, 
                phono3py_yaml=args.phono3py_yaml, 
                st_dict=st_dict, 
                supercell_matrix=supercell_matrix,
                n_samples=args.n_samples,
                displacements=0.03)

