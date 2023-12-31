#!/usr/bin/env python
import numpy as np
from collections import Counter

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

def st_dict_to_phonopy_cell(st_dict):

    ph_cell = PhonopyAtoms(st_dict['elements'],
                           cell=st_dict['axis'].T,
                           scaled_positions=st_dict['positions'].T)
    return ph_cell

def phonopy_cell_to_st_dict(ph_cell):

    st_dict = dict()
    st_dict['axis'] = ph_cell.cell.T
    st_dict['positions'] = ph_cell.scaled_positions.T
    st_dict['elements'] = ph_cell.symbols
    st_dict['volume'] = ph_cell.volume

    elements_uniq = sorted(set(st_dict['elements']), 
                           key=st_dict['elements'].index)
    elements_count = Counter(st_dict['elements'])
    st_dict['n_atoms'] = [elements_count[ele] for ele in elements_uniq]
    st_dict['types'] = [i for i, n in enumerate(st_dict['n_atoms']) 
                          for _ in range(n)]

    return st_dict

def phonopy_supercell(st_dict, 
                      supercell_matrix=None, 
                      supercell_diag=None,
                      return_phonopy=True):

    if supercell_diag is not None:
        supercell_matrix = np.diag(supercell_diag) 

    unitcell = st_dict_to_phonopy_cell(st_dict)
    supercell = Phonopy(unitcell, supercell_matrix).supercell
    if return_phonopy:
        return supercell

    supercell_dict = phonopy_cell_to_st_dict(supercell)
    supercell_dict['supercell_matrix'] = supercell_matrix
    return supercell_dict



