#!/usr/bin/env python
import argparse
import os

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.vasp_utils import write_poscar_file


def write_structures(st_dicts, base_info, output_dir="poscars"):

    os.makedirs(output_dir, exist_ok=True)
    f = open("polymlp_str_samples.yaml", "w")
    print("prototypes:", file=f)
    for base_dict in base_info:
        print("- id:             ", base_dict["id"], file=f)
        print("  supercell_size: ", base_dict["size"], file=f)
        print("  n_atoms:        ", base_dict["n_atoms"], file=f)

    print("structures:", file=f)
    for i, st in enumerate(st_dicts):
        filename = "poscars/poscar-" + str(i + 1).zfill(5)
        header = "pypolymlp: random-" + str(i + 1).zfill(5)
        write_poscar_file(st, filename=filename, header=header)
        print("- id:", str(i + 1).zfill(5), file=f)
        print("  base:", st["base"], file=f)
        print("  mode:", st["mode"], file=f)

    f.close()


def set_structure_id(st_dicts, poscar, mode):
    for st in st_dicts:
        st["base"] = poscar
        st["mode"] = mode
    return st_dicts


class StructureGenerator:

    def __init__(self, st_dict, natom_lb=48, natom_ub=150):

        self.unitcell = st_dict

        if natom_lb > natom_ub:
            raise ValueError("natom_lb > n_atom_ub in StructureGenerator")
        self.natom_lb = natom_lb
        self.natom_ub = natom_ub

        self.supercell = self.__set_supercell()

    def __set_supercell(self):

        self.size = self.__find_supercell_size_nearly_isotropic()
        self.supercell = supercell_diagonal(self.unitcell, self.size)
        self.supercell["axis_inv"] = np.linalg.inv(self.supercell["axis"])
        return self.supercell

    def __find_supercell_size_nearly_isotropic(self):

        axis = self.unitcell["axis"]
        total_n_atoms = sum(self.unitcell["n_atoms"])

        len_axis = [np.linalg.norm(axis[:, i]) for i in range(3)]
        ratio1 = len_axis[0] / len_axis[1]
        ratio2 = len_axis[0] / len_axis[2]
        ratio = np.array([1, ratio1, ratio2])

        cand = np.arange(1, 11)
        size = [1, 1, 1]
        for c in cand:
            size_trial = np.maximum(np.round(ratio * c).astype(int), [1, 1, 1])
            n_total = total_n_atoms * np.prod(size_trial)
            if n_total >= self.natom_lb:
                if n_total <= self.natom_ub:
                    size = size_trial
                break
            size = size_trial
        return size

    def random_single_structure(self, disp, vol_ratio=1.0):

        cell = self.supercell
        total_n_atoms = cell["positions"].shape[1]
        axis_ratio = pow(vol_ratio, 1.0 / 3.0)

        axis_add = (np.random.rand(3, 3) * 2.0 - 1) * disp
        positions_add = (np.random.rand(3, total_n_atoms) * 2.0 - 1) * disp
        positions_add = cell["axis_inv"] @ positions_add

        str_rand = dict()
        str_rand["axis"] = cell["axis"] * axis_ratio + axis_add
        str_rand["positions"] = cell["positions"] + positions_add
        str_rand["n_atoms"] = cell["n_atoms"]
        str_rand["elements"] = cell["elements"]
        str_rand["types"] = cell["types"]
        return str_rand

    def random_structure_algo2(self, n_str=10, max_disp=0.3, vol_ratio=1.0):
        """disp = [(i + 1) / n_str] * max_disp"""
        st_array = []
        for i in range(n_str):
            disp = (i + 1) * max_disp / float(n_str)
            str_rand = self.random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)

        return st_array

    def random_structure(self, n_str=10, max_disp=1.0, vol_ratio=1.0):
        """disp = max([(i + 1) / n_str]**3 * max_disp, 0.01)"""
        st_array = []
        for i in range(n_str):
            disp_function1 = 0.01
            disp_function2 = pow((i + 1) / float(n_str), 3.0) * max_disp
            disp = max(disp_function1, disp_function2)
            str_rand = self.random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)

        return st_array

    def sample_density(self, n_str=10, disp=0.2, vol_lb=1.1, vol_ub=4.0):

        st_array = []
        for vol_ratio in np.linspace(vol_lb, vol_ub, n_str):
            str_rand = self.random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)
        return st_array

    def print_size(self):
        print("  supercell size:      ", list(self.size))
        print("  n_atoms (supercell): ", list(self.supercell["n_atoms"]))


def run_strgen(args):

    sampled_structures = []
    base_info = []
    for poscar in args.poscars:
        unitcell = Poscar(poscar).get_structure()
        gen = StructureGenerator(unitcell, natom_ub=args.max_natom)

        base_dict = dict()
        base_dict["id"] = poscar
        base_dict["size"] = list(gen.size)
        base_dict["n_atoms"] = list(gen.supercell["n_atoms"])
        base_info.append(base_dict)

        print("-----------------------")
        print("-", poscar)
        gen.print_size()

        if args.n_str is not None:
            st_dicts = gen.random_structure(
                n_str=args.n_str, max_disp=args.max_disp, vol_ratio=1.0
            )
            st_dicts = set_structure_id(st_dicts, poscar, "standard")
            sampled_structures.extend(st_dicts)
        if args.low_density is not None:
            st_dicts = gen.sample_density(
                n_str=args.low_density,
                disp=args.density_mode_disp,
                vol_lb=1.1,
                vol_ub=4.0,
            )
            st_dicts = set_structure_id(st_dicts, poscar, "low density")
            sampled_structures.extend(st_dicts)
        if args.high_density is not None:
            st_dicts = gen.sample_density(
                n_str=args.low_density,
                disp=args.density_mode_disp,
                vol_lb=0.6,
                vol_ub=0.9,
            )
            st_dicts = set_structure_id(st_dicts, poscar, "high density")
            sampled_structures.extend(st_dicts)

    write_structures(sampled_structures, base_info)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscars",
        type=str,
        nargs="*",
        required=True,
        help="Initial structures in POSCAR format",
    )
    parser.add_argument(
        "--max_natom",
        type=int,
        default=150,
        help="Maximum number of atoms in structures",
    )
    parser.add_argument(
        "-n",
        "--n_str",
        type=int,
        default=None,
        help="Number of structures sampled " + "using a standard algorithm",
    )
    parser.add_argument(
        "-d",
        "--max_disp",
        type=float,
        default=1.5,
        help="Maximum random number for generating " "atomic displacements",
    )

    parser.add_argument(
        "--low_density",
        type=int,
        default=None,
        help="Number of structures for low density mode.",
    )
    parser.add_argument(
        "--high_density",
        type=int,
        default=None,
        help="Number of structures for high density mode.",
    )
    parser.add_argument(
        "--density_mode_disp",
        type=float,
        default=0.2,
        help="Maximum random number for generating atomic "
        "displacements in low and high density modes",
    )

    args = parser.parse_args()

    run_strgen(args)
