#!/usr/bin/env python
import numpy as np
from pyclupan.common.supercell import Supercell

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import Poscar


def make_supercell(unitcell, hnf):

    sup = Supercell(
        axis=unitcell["axis"],
        positions=unitcell["positions"],
        n_atoms=unitcell["n_atoms"],
        hnf=hnf,
    )
    sup.construct_supercell()
    st = sup.get_supercell()

    element_list = []
    idx = 0
    for n in unitcell["n_atoms"]:
        element_list.append(unitcell["elements"][idx])
        idx += n

    st_dict_sup = dict()
    st_dict_sup["axis"] = st.axis
    st_dict_sup["positions"] = st.positions
    st_dict_sup["n_atoms"] = st.n_atoms
    st_dict_sup["types"] = st.types
    st_dict_sup["volume"] = st.volume
    st_dict_sup["elements"] = [
        element_list[i] for i, n in enumerate(st.n_atoms) for _ in range(n)
    ]
    return st_dict_sup


unitcell = Poscar("POSCAR").get_structure()
prop = Properties(pot="polymlp.lammps")

expansions = np.array(
    [
        np.eye(3),
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [-1, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-1, 1, 0], [-1, -1, 1]],
        [[1, 0, 0], [-3, 1, 0], [3, 3, 1]],
        [[1, 0, 0], [3, 1, 0], [3, 3, 1]],
        [[1, 0, 0], [-3, 1, 0], [-3, -3, 1]],
        [[1, 0, 0], [5, 1, 0], [5, 5, 1]],
        [[1, 0, 0], [-5, 1, 0], [-5, -5, 1]],
        [[1, 0, 0], [10, 1, 0], [10, 10, 1]],
        [[1, 0, 0], [-10, 1, 0], [-10, -10, 1]],
        [[1, 3, 3], [0, 1, 3], [0, 0, 1]],
        [[1, 4, 4], [0, 1, 4], [0, 0, 1]],
        [[1, 5, 5], [0, 1, 5], [0, 0, 1]],
        [[1, -5, -5], [0, 1, -5], [0, 0, 1]],
        [[1, 10, 10], [0, 1, 10], [0, 0, 1]],
        [[1, -10, -10], [0, 1, -10], [0, 0, 1]],
        [[1, 0, 0], [5, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-5, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [10, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-10, 1, 0], [0, 0, 1]],
    ]
)

for hnf in expansions:
    st_rev = make_supercell(unitcell, hnf)
    energy, forces, stresses = prop.eval(st_rev)
    print(energy, hnf[0], hnf[1], hnf[2])


unitcell = Poscar("POSCAR2").get_structure()
print("------")
for hnf in expansions:
    st_rev = make_supercell(unitcell, hnf)
    energy, forces, stresses = prop.eval(st_rev)
    print(energy, hnf[0], hnf[1], hnf[2])

unitcell = Poscar("POSCAR3").get_structure()
print("------")
for hnf in expansions:
    st_rev = make_supercell(unitcell, hnf)
    energy, forces, stresses = prop.eval(st_rev)
    print(energy, hnf[0], hnf[1], hnf[2])
