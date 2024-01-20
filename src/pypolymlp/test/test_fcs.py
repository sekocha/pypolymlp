#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import (
        phonopy_supercell,
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)

from pypolymlp.core.displacements import generate_random_const_displacements

from pypolymlp.calculator.compute_properties import compute_properties
from pypolymlp.calculator.compute_fcs import recover_fc2, recover_fc3
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3

def compute_fcs_from_dataset(pot, supercell, disps, st_dicts):

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
    t1 = time.time()
    fc2_basis = FCBasisSetO2(supercell, use_mkl=False).run()
    compress_mat_fc2 = fc2_basis.compression_matrix
    compress_eigvecs_fc2 = fc2_basis.basis_set

    #fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    fc3_basis = FCBasisSetO3(supercell, use_mkl=True).run()
    compress_mat_fc3 = fc3_basis.compression_matrix
    compress_eigvecs_fc3 = fc3_basis.basis_set
    t2 = time.time()
    print(' elapsed time (basis sets for fc2 and fc3) =', t2-t1)

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


def compute_fcs_from_structure(pot, 
                               unitcell_dict=None, 
                               supercell_dict=None, 
                               supercell_matrix=None,
                               n_samples=100,
                               displacements=0.03):

    if supercell_dict is not None:
        supercell = st_dict_to_phonopy_cell(supercell_dict)
    elif unitcell_dict is not None:
        supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
        supercell_dict = phonopy_cell_to_st_dict(supercell)

    disps, st_dicts = generate_random_const_displacements(
            supercell_dict,
            n_samples=n_samples,
            displacements=displacements
    )
    compute_fcs_from_dataset(pot, supercell, disps, st_dicts)



def compute_fc_basis(supercell):

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


