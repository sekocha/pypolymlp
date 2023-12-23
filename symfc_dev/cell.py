#!/usr/bin/env python
import numpy as np
import argparse

from lammps_api.common.structure_poscar import PoscarFormat
from lammps_api.property.phonon import LammpsPhonon

from phonopy.structure.atoms import PhonopyAtoms

def poscar_to_supercell(poscar, supercell_mat):

    p = PoscarFormat().parse_file(filename=poscar)
    lmp_phonon = LammpsPhonon(pot=None, mlp=True, log=False)
    lmp_phonon.setup_from_file(poscar_st=p, 
                               supercell_matrix=supercell_mat,
                               elements_from_pot=False)
    return (lmp_phonon.unitcell, lmp_phonon.supercell)
 
def st_dict_to_phonony(st_dict):
    st_phonopy = PhonopyAtoms(st_dict['elements'],
                              cell=st_dict['axis'].T,
                              scaled_positions=st_dict['positions'].T)
    return st_phonopy
