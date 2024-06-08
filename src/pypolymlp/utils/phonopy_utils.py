#!/usr/bin/env python
from collections import Counter

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from pypolymlp.calculator.properties import Properties


def st_dict_to_phonopy_cell(st_dict):

    ph_cell = PhonopyAtoms(
        symbols=st_dict["elements"],
        cell=st_dict["axis"].T,
        scaled_positions=st_dict["positions"].T,
    )
    return ph_cell


def phonopy_cell_to_st_dict(ph_cell):

    st_dict = dict()
    st_dict["axis"] = ph_cell.cell.T
    st_dict["positions"] = ph_cell.scaled_positions.T
    st_dict["elements"] = ph_cell.symbols
    st_dict["volume"] = ph_cell.volume

    elements_uniq = sorted(set(st_dict["elements"]), key=st_dict["elements"].index)
    elements_count = Counter(st_dict["elements"])
    st_dict["n_atoms"] = [elements_count[ele] for ele in elements_uniq]
    st_dict["types"] = [i for i, n in enumerate(st_dict["n_atoms"]) for _ in range(n)]

    return st_dict


def phonopy_supercell(
    st_dict, supercell_matrix=None, supercell_diag=None, return_phonopy=True
):

    if supercell_diag is not None:
        supercell_matrix = np.diag(supercell_diag)

    unitcell = st_dict_to_phonopy_cell(st_dict)
    supercell = Phonopy(unitcell, supercell_matrix).supercell
    if return_phonopy:
        return supercell

    supercell_dict = phonopy_cell_to_st_dict(supercell)
    supercell_dict["supercell_matrix"] = supercell_matrix
    return supercell_dict


def compute_forces_phonopy_displacements(
    ph: Phonopy, pot="polymlp.lammps", distance=0.01
):
    """Compute forces using phonopy object and polymlp.
    Return
    ------
    forces: (n_str, n_atom, 3)
    """
    prop = Properties(pot=pot)

    ph.generate_displacements(distance=distance)
    supercells = ph.supercells_with_displacements
    st_dicts = [phonopy_cell_to_st_dict(cell) for cell in supercells]
    """ forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)"""
    _, forces, _ = prop.eval_multiple(st_dicts)
    forces = np.array(forces).transpose((0, 2, 1))
    return forces
