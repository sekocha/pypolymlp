#!/usr/bin/env python
import copy

import numpy as np
import spglib
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1

from pypolymlp.utils.phonopy_utils import st_dict_to_phonopy_cell

# from math import sqrt


"""Atomic Coodinates"""


def basis_cartesian_to_fractional_coordinates(basis_c, unitcell):

    n_basis = basis_c.shape[1]
    n_atom = len(unitcell["elements"])
    unitcell["axis_inv"] = np.linalg.inv(unitcell["axis"])

    basis_c = np.array([b.reshape((n_atom, 3)) for b in basis_c.T])
    basis_c = basis_c.transpose((2, 1, 0))
    basis_c = basis_c.reshape(3, -1)
    basis_f = unitcell["axis_inv"] @ basis_c
    basis_f = basis_f.reshape((3, n_atom, n_basis))
    basis_f = basis_f.transpose((1, 0, 2)).reshape(-1, n_basis)
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


"""Cell Parameters"""


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def basis_cell_metric(cell):

    cell_copy = copy.deepcopy(cell)
    cell_ph = st_dict_to_phonopy_cell(cell_copy)
    cell, scaled_positions, _ = spglib.standardize_cell(cell_ph)
    spg_info = spglib.get_symmetry_dataset(cell_ph)
    spg_num = spg_info["number"]

    cell_copy["axis"] = cell.T
    cell_copy["positions"] = scaled_positions.T

    """basis (row): In the order of Gaa, Gbb, Gcc, Gab, Gbc, Gca"""
    if spg_num >= 195:
        print("Crystal system: Cubic")
        basis = np.zeros((6, 1))
        basis[:, 0] = normalize_vector([1, 1, 1, 0, 0, 0])
    elif spg_num >= 168 and spg_num <= 194:
        print("Crystal system: Hexagonal")
        basis = np.zeros((6, 2))
        basis[:, 0] = normalize_vector([1, 1, 0, -0.5, 0, 0])
        basis[2, 1] = 1.0
    elif spg_num >= 143 and spg_num <= 167:
        if "P" in spg_info["international"]:
            print("Crystal system: Trigonal (Hexagonal)")
            basis = np.zeros((6, 2))
            basis[:, 0] = normalize_vector([1, 1, 0, -0.5, 0, 0])
            basis[2, 1] = 1.0
        else:
            print("Crystal system: Trigonal (Rhombohedral)")
            basis = np.zeros((6, 2))
            basis[:, 0] = normalize_vector([1, 1, 0, -0.5, 0, 0])
            basis[2, 1] = 1.0
    elif spg_num >= 75 and spg_num <= 142:
        print("Crystal system: Tetragonal")
        basis = np.zeros((6, 2))
        basis[:, 0] = normalize_vector([1, 1, 0, 0, 0, 0])
        basis[2, 1] = 1.0
    elif spg_num >= 16 and spg_num <= 74:
        print("Crystal system: Orthorhombic")
        basis = np.zeros((6, 3))
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        basis[2, 2] = 1.0
    elif spg_num >= 3 and spg_num <= 15:
        print("Crystal system: Monoclinic")
        basis = np.zeros((6, 4))
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        basis[2, 2] = 1.0
        basis[5, 3] = 1.0
    else:
        print("Crystal system: Triclinic")
        basis = np.eye(6)

    return basis, cell_copy


if __name__ == "__main__":

    import argparse

    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--pot", type=str, default="polymlp.lammps", help="polymlp file"
    )
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    #    basis_c = construct_basis_cartesian(unitcell)
    #    print('Basis Cartesian:')
    #    print(basis_c)

    basis, cell = basis_cell_metric(unitcell)
    print(basis)
    print(cell)

"""
# def basis_cell(cell, tol=1e-10):
#
#    cell_copy = copy.deepcopy(cell)
##    cell_copy['positions'] = np.array([[0,0,0]]).T
##    cell_copy['n_atoms'] = [1]
##    cell_copy['types'] = [0]
##    cell_copy['elements'] = [cell['elements'][0]]
#
#
#    cell_ph = st_dict_to_phonopy_cell(cell_copy)
#    print(spglib.standardize_cell(cell_ph))
#
#    spglib_obj = spglib.get_symmetry(cell_ph)
#    rotations = spglib_obj['rotations']
#    translations = spglib_obj['translations']
#
#    print(spglib.get_symmetry_dataset(cell_ph))
#    print(spglib_obj['pointgroup_symbol'])
#
#    #rotations_lattice = []
#    #for r, t in zip(rotations, translations):
#    #    if np.allclose(np.zeros(3), t):
#    #        rotations_lattice.append(r)
#
#    #axis_inv = np.linalg.inv(cell['axis'])
#    #rotations_lattice = [rot for rot in rotations
#    #                     if len(np.nonzero(rot)[0]) == 3 ]
#
#    rotations_lattice = rotations
#    proj = np.zeros((9,9))
#    for rot in rotations_lattice:
#        rep = np.zeros((9,9))
#        for i,j in itertools.product(range(3),range(3)):
#            row = 3 * i + j
#            for k,l in itertools.product(range(3),range(3)):
#                col = 3 * k + l
#                rep[row,col] = rot[k,i] * rot[l,j]
#        proj += rep
#    proj /= len(rotations_lattice)
#
#
##    for rot in rotations_lattice:
##        #proj += from_rotation_to_representation(rot, cell['axis'], axis_inv)
##        proj += np.kron(rot, rot)
##    proj /= len(rotations_lattice)
#
#    C = np.zeros((9,6))
#    C[0,0] = 1
#    C[4,1] = 1
#    C[8,2] = 1
#    C[1,3] = 1/sqrt(2)
#    C[3,3] = 1/sqrt(2)
#    C[2,4] = 1/sqrt(2)
#    C[6,4] = 1/sqrt(2)
#    C[5,5] = 1/sqrt(2)
#    C[7,5] = 1/sqrt(2)
#    proj2 = C @ C.T
#    proj = proj @ proj2
#
#    eigvals, eigvecs = np.linalg.eigh(proj)
#    nonzero = np.isclose(eigvals,1.0)
#    print('Eigenvalues:')
#    print(eigvals)
#    print(eigvecs[:,nonzero])
#    print(eigvecs[:,-2])
#
#
#
"""
