#!/usr/bin/env python
import glob

from pypolymlp.core.interface_vasp import Vasprun
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies

path_prototype1 = (
    "/home/seko/home-nas0/mlip-dft/1-unary" + "/2024-organized/1-ideal/1-prototypes/"
)

"""for elemental system"""


def __read_structure_types1():

    f = open(path_prototype1 + "structure_type")
    lines_st = f.readlines()
    f.close()

    structure_types_dict = dict()
    for line in lines_st:
        split = line.split()
        num = split[2].replace(";", "")
        st_type = split[5]
        structure_types_dict[num] = st_type
    return structure_types_dict


"""for elemental system"""


def get_icsd_data1(elements, path_vasp):

    structure_types = __read_structure_types1()

    files = sorted(glob.glob(path_vasp + "/*/vasprun.xml"))
    atom_e, elements = get_atomic_energies(elements=elements, functional="PBE")
    atom_e = dict(zip(elements, atom_e))

    icsd_data = dict()
    for file1 in files:
        v = Vasprun(file1)
        e, structure = v.energy, v.structure
        e -= sum([float(atom_e[ele]) for ele in structure.elements])
        e /= sum(structure.n_atoms)

        icsd_id = file1.split("/")[-2]
        key = "icsd-" + icsd_id + "-[" + structure_types[icsd_id] + "]"
        icsd_data[key] = {
            "DFT_energy": e,
            "MLP_energy": None,
            "structure": structure,
            "icsd_id": icsd_id,
        }

    return icsd_data
