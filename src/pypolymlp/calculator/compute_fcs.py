#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.yaml_utils import load_cells
from pypolymlp.utils.phonopy_utils import (
        phonopy_supercell,
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)
from pypolymlp.core.displacements import generate_random_const_displacements

from pypolymlp.calculator.properties import Properties
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
from symfc.solvers.solver_O2O3 import run_solver_sparse_O2O3

from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

def recover_fc2(coefs, compress_mat, compress_eigvecs, N):
    ''' if using full compression_matrix
    fc2 = compress_eigvecs @ coefs
    fc2 = (compress_mat @ fc2).reshape((N,N,3,3))
    '''
    n_a = compress_mat.shape[0] // (9*N)
    fc2 = compress_eigvecs @ coefs
    fc2 = (compress_mat @ fc2).reshape((n_a,N,3,3))
    return fc2

def recover_fc3(coefs, compress_mat, compress_eigvecs, N):
    ''' if using full compression_matrix
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((N,N,N,3,3,3))
    '''
    n_a = compress_mat.shape[0] // (27*(N**2))
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((n_a,N,N,3,3,3))
    return fc3

def compute_fcs_from_dataset(st_dicts, disps, supercell, 
                             pot=None, params_dict=None, coeffs=None):

    ''' disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
    disps = disps.transpose((0,2,1)) 

    t1 = time.time()
    ''' forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)'''
    prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)
    _, forces, _ = prop.eval_multiple(st_dicts)
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

    fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
    #fc3_basis = FCBasisSetO3(supercell, use_mkl=True).run()
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
    #write_fc2_to_hdf5(fc2)
    write_fc3_to_hdf5(fc3)


def compute_fcs_from_structure(pot=None, 
                               params_dict=None,
                               coeffs=None,
                               unitcell_dict=None, 
                               supercell_matrix=None,
                               supercell_dict=None, 
                               n_samples=100,
                               displacements=0.03,
                               is_plusminus=False):

    if supercell_dict is not None:
        supercell = st_dict_to_phonopy_cell(supercell_dict)
    elif unitcell_dict is not None:
        supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
        supercell_dict = phonopy_cell_to_st_dict(supercell)
    else:
        raise ValueError('(unitcell_dict, supercell_matrix) '
                         'or supercell_dict is requried.')

    disps, st_dicts = generate_random_const_displacements(
            supercell_dict,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
    )
    compute_fcs_from_dataset(st_dicts, disps, supercell, 
                             pot=pot, params_dict=params_dict, coeffs=coeffs)


def compute_fcs_phono3py_dataset(pot=None, 
                                 params_dict=None,
                                 coeffs=None,
                                 phono3py_yaml=None, 
                                 use_phonon_dataset=False,
                                 n_samples=None,
                                 displacements=0.03,
                                 is_plusminus=False):

    supercell, disps, st_dicts = parse_phono3py_yaml_fcs(
            phono3py_yaml,
            use_phonon_dataset=use_phonon_dataset
    )

    if n_samples is not None:
        supercell_dict = phonopy_cell_to_st_dict(supercell)
        disps, st_dicts = generate_random_const_displacements(
                supercell_dict,
                n_samples=n_samples,
                displacements=displacements,
                is_plusminus=is_plusminus,
        )

    compute_fcs_from_dataset(st_dicts, disps, supercell, 
                             pot=pot, params_dict=params_dict, coeffs=coeffs)


