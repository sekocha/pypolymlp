#!/usr/bin/env python 
import numpy as np
import copy
from math import sqrt

from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1
from pypolymlp.utils.phonopy_utils import st_dict_to_phonopy_cell

import spglib


'''Atomic Coodinates'''
def basis_cartesian_to_fractional_coordinates(basis_c, unitcell):

    n_basis = basis_c.shape[1]
    n_atom = len(unitcell['elements'])
    unitcell['axis_inv'] = np.linalg.inv(unitcell['axis'])

    basis_c = np.array([b.reshape((n_atom, 3)) for b in basis_c.T])
    basis_c = basis_c.transpose((2,1,0))
    basis_c = basis_c.reshape(3, -1)
    basis_f = unitcell['axis_inv'] @ basis_c
    basis_f = basis_f.reshape((3,n_atom,n_basis))
    basis_f = basis_f.transpose((1,0,2)).reshape(-1, n_basis)
    basis_f, _, _ = np.linalg.svd(basis_f, full_matrices=False)
    return basis_f


def construct_basis_cartesian(cell):

    cell_ph = st_dict_to_phonopy_cell(cell)
    fc_basis = FCBasisSetO1(cell_ph).run()
    return fc_basis.full_basis_set.toarray()


def construct_basis_fractional_coordinates(cell):

    basis_c = construct_basis_cartesian(cell)
    basis_f = basis_cartesian_to_fractional_coordinates(basis_c, cell)
    return basis_f


'''Cell Parameters'''
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def basis_cell(cell):

    cell_copy = copy.deepcopy(cell)
    cell_ph = st_dict_to_phonopy_cell(cell_copy)
    lattice, scaled_positions, _ = spglib.standardize_cell(cell_ph)

    cell_copy['axis'] = lattice.T
    cell_copy['positions'] = scaled_positions.T

    '''basis (row): In the order of ax, bx, cx, ay, by, cy, az, bz, cz'''
    spg_info = spglib.get_symmetry_dataset(cell_ph)
    spg_num = spg_info['number']

    if spg_num >= 195:
        print('Crystal system: Cubic')
        basis = np.zeros((9,1))
        basis[:,0] = normalize_vector([1,0,0,0,1,0,0,0,1])
    elif spg_num >= 168 and spg_num <= 194:
        print('Crystal system: Hexagonal')
        basis = np.zeros((9,2))
        basis[:,0] = normalize_vector([1,-0.5,0,0,sqrt(3)/2,0,0,0,0])
        basis[8,1] = 1.0
    elif spg_num >= 143 and spg_num <= 167:
        if 'P' in spg_info['international']:
            print('Crystal system: Trigonal (Hexagonal)')
            basis = np.zeros((9,2))
            basis[:,0] = normalize_vector([1,-0.5,0,0,sqrt(3)/2,0,0,0,0])
            basis[8,1] = 1.0
        else:
            print('Crystal system: Trigonal (Rhombohedral)')
            basis = np.zeros((9,2))
            basis[:,0] = normalize_vector([1,-0.5,0,0,sqrt(3)/2,0,0,0,0])
            basis[8,1] = 1.0
    elif spg_num >= 75 and spg_num <= 142:
        print('Crystal system: Tetragonal')
        basis = np.zeros((9,2))
        basis[:,0] = normalize_vector([1,0,0,0,1,0,0,0,0])
        basis[8,1] = 1.0
    elif spg_num >= 16 and spg_num <= 74:
        print('Crystal system: Orthorhombic')
        basis = np.zeros((9,3))
        basis[0,0] = 1.0
        basis[4,1] = 1.0
        basis[8,2] = 1.0
    elif spg_num >= 3 and spg_num <= 15:
        print('Crystal system: Monoclinic')
        basis = np.zeros((9,4))
        basis[0,0] = 1.0
        basis[4,1] = 1.0
        basis[8,2] = 1.0
        basis[2,3] = 1.0
    else:
        print('Crystal system: Triclinic')
        basis = np.eye(9)

    return basis, cell_copy



if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        type=str,
                        default=None,
                        help='poscar file')
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    basis, cell = basis_cell(unitcell)

