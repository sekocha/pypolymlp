#!/usr/bin/env python 
import numpy as np
import itertools
import copy

from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1
from pypolymlp.utils.phonopy_utils import st_dict_to_phonopy_cell

import spglib
from phonopy.structure.cells import compute_all_sg_permutations

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

#def from_rotation_to_representation(rot, axis, axis_inv):
#    rep = np.zeros((9,9))
#    for (i,j) in itertools.product(range(3),range(3)):
#        rot2 = np.dot(axis, np.dot(rot, axis_inv))
#        rep[3*i:3*i+3,3*j:3*j+3] = rot[i,j] * rot2
#    return rep

#def from_rotation_to_representation(rot, axis, axis_inv):
#    rot2 = axis @ rot @ axis_inv
#    rep = np.zeros((9,9))
#    for (i,j) in itertools.product(range(3),range(3)):
#        rep[3*i:3*i+3,3*j:3*j+3] += rot[j,i] * rot2.T
#    return rep

#def from_rotation_to_representation(rot, axis, axis_inv):

def basis_cell(cell, tol=1e-10):

    cell_ph = st_dict_to_phonopy_cell(cell)
    spglib_obj = spglib.get_symmetry(cell_ph)
    rotations = spglib_obj['rotations']

    axis_inv = np.linalg.inv(cell['axis'])
    rotations_lattice = [rot for rot in rotations 
                         if len(np.nonzero(rot)[0]) == 3]
    #rotations_lattice = rotations

    proj = np.zeros((9,9))
    for rot in rotations_lattice:
        #proj += from_rotation_to_representation(rot, cell['axis'], axis_inv)
        proj += np.kron(rot, rot)
    proj /= len(rotations_lattice)

#    C = np.zeros((9,6))
#    for j, i in enumerate([0,1,2,4,5,8]):
#        C[i,j] = 1.0
#    proj2 = C @ C.T
#    print(proj2)
#    print(proj @ proj2 - proj2 @ proj)
#    #proj @ proj2

#    ids_zero = np.array([3])
#    proj[:,ids_zero] = 0
#    proj[ids_zero] = 0

    eigvals, eigvecs = np.linalg.eigh(proj)
    nonzero = np.isclose(eigvals,1.0)
    print('Eigenvalues:')
    print(eigvals)
    print(eigvecs[:,nonzero])


if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        type=str,
                        default=None,
                        help='poscar file')
    parser.add_argument('--pot',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
#    basis_c = construct_basis_cartesian(unitcell)
#    print('Basis Cartesian:')
#    print(basis_c)

    basis_cell(unitcell)


