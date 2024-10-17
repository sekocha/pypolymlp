#!/usr/bin/env python
from math import sqrt

import numpy as np
import spglib
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell


def basis_cartesian_to_fractional_coordinates(
    basis_c: np.ndarray,
    unitcell: PolymlpStructure,
) -> np.ndarray:
    """Convert basis set in Cartesian coord. to basis set in fractional coordinates."""
    n_basis = basis_c.shape[1]
    n_atom = len(unitcell.elements)
    unitcell.axis_inv = np.linalg.inv(unitcell.axis)

    basis_c = np.array([b.reshape((n_atom, 3)) for b in basis_c.T])
    basis_c = basis_c.transpose((2, 1, 0))
    basis_c = basis_c.reshape(3, -1)
    basis_f = unitcell.axis_inv @ basis_c
    basis_f = basis_f.reshape((3, n_atom, n_basis))
    basis_f = basis_f.transpose((1, 0, 2)).reshape(-1, n_basis)
    basis_f, _, _ = np.linalg.svd(basis_f, full_matrices=False)
    return basis_f


def construct_basis_cartesian(cell: PolymlpStructure) -> np.ndarray:
    """Generate a basis set for atomic positions in Cartesian coordinates."""
    cell_ph = structure_to_phonopy_cell(cell)
    try:
        fc_basis = FCBasisSetO1(cell_ph).run()
    except ValueError:
        return None
    return fc_basis.full_basis_set.toarray()


def construct_basis_fractional_coordinates(cell: PolymlpStructure) -> np.ndarray:
    """Generate a basis set for atomic positions in fractional coordinates."""
    basis_c = construct_basis_cartesian(cell)
    if basis_c is None:
        return None
    elif basis_c.size == 0:
        return None
    basis_f = basis_cartesian_to_fractional_coordinates(basis_c, cell)
    return basis_f


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def standardize_cell(cell: PolymlpStructure) -> PolymlpStructure:
    """Standardize cell."""
    cell_ph = structure_to_phonopy_cell(cell)

    map_numbers, map_elements = dict(), dict()
    for n, t, e in zip(cell_ph.numbers, cell.types, cell.elements):
        map_numbers[n] = t
        map_elements[t] = e

    lattice, scaled_positions, numbers = spglib.standardize_cell(
        (cell_ph.cell, cell_ph.scaled_positions, cell_ph.numbers),
        to_primitive=False,
    )
    types = [map_numbers[n] for n in numbers]

    scaled_positions_reorder, types_reorder = [], []
    n_atoms = []
    for i in range(max(types) + 1):
        ids = np.array(types) == i
        scaled_positions_reorder.extend(scaled_positions[ids])
        types_reorder.extend(np.array(types)[ids])
        n_atoms.append(np.count_nonzero(ids))
    scaled_positions_reorder = np.array(scaled_positions_reorder)
    elements = [map_elements[t] for t in types]

    cell_standardized = PolymlpStructure(
        axis=lattice.T,
        positions=scaled_positions_reorder.T,
        n_atoms=n_atoms,
        elements=elements,
        types=types,
    )
    return cell_standardized


def basis_cell(cell: PolymlpStructure) -> tuple[np.ndarray, PolymlpStructure]:
    """Generate a basis set for axis matrix."""

    cell_copy = standardize_cell(cell)
    cell_ph = structure_to_phonopy_cell(cell_copy)

    """basis (row): In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
    spg_info = spglib.get_symmetry_dataset(
        (cell_ph.cell, cell_ph.scaled_positions, cell_ph.numbers),
    )
    spg_num = spg_info["number"]
    print("Space group:", spg_info["international"], spg_num)

    if spg_num >= 195:
        print("Crystal system: Cubic")
        basis = np.zeros((9, 1))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 1])
    elif spg_num >= 168 and spg_num <= 194:
        print("Crystal system: Hexagonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, sqrt(3) / 2, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif spg_num >= 143 and spg_num <= 167:
        if "P" in spg_info["international"]:
            print("Crystal system: Trigonal (Hexagonal)")
            basis = np.zeros((9, 2))
            basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, sqrt(3) / 2, 0, 0, 0, 0])
            basis[8, 1] = 1.0
        else:
            print("Crystal system: Trigonal (Rhombohedral)")
            basis = np.zeros((9, 2))
            basis[:, 0] = _normalize_vector([1, -0.5, 0, 0, sqrt(3) / 2, 0, 0, 0, 0])
            basis[8, 1] = 1.0
    elif spg_num >= 75 and spg_num <= 142:
        print("Crystal system: Tetragonal")
        basis = np.zeros((9, 2))
        basis[:, 0] = _normalize_vector([1, 0, 0, 0, 1, 0, 0, 0, 0])
        basis[8, 1] = 1.0
    elif spg_num >= 16 and spg_num <= 74:
        print("Crystal system: Orthorhombic")
        basis = np.zeros((9, 3))
        basis[0, 0] = 1.0
        basis[4, 1] = 1.0
        basis[8, 2] = 1.0
    elif spg_num >= 3 and spg_num <= 15:
        print("Crystal system: Monoclinic")
        basis = np.zeros((9, 4))
        basis[0, 0] = 1.0
        basis[4, 1] = 1.0
        basis[8, 2] = 1.0
        basis[2, 3] = 1.0
    else:
        print("Crystal system: Triclinic")
        basis = np.eye(9)

    return basis, cell_copy


if __name__ == "__main__":

    import argparse

    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).structure
    basis, cell = basis_cell(unitcell)
