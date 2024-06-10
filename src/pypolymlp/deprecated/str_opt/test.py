#!/usr/bin/env python
import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1

from pypolymlp.core.interface_vasp import Poscar

st_dict = Poscar("POSCAR").get_structure()

cell_ph = PhonopyAtoms(
    symbols=st_dict["elements"],
    cell=st_dict["axis"].T,
    scaled_positions=st_dict["positions"].T,
)
print(cell_ph.cell)
print(cell_ph.scaled_positions)
print(cell_ph.symbols)

fc_basis = FCBasisSetO1(cell_ph).run()

basis = fc_basis.full_basis_set.toarray()
print(basis)
print(basis.shape)


# cell2 = (cell_ph.cell, cell_ph.scaled_positions, cell_ph.numbers)
# sym = spglib.get_symmetry(cell=cell2, symprec=1e-3)
# spg = spglib.get_spacegroup(cell=cell2, symprec=1e-3)

sym = spglib.get_symmetry(cell_ph, symprec=1e-3)
spg = spglib.get_spacegroup(cell_ph, symprec=1e-3)
print(sym)
print(spg)

#
