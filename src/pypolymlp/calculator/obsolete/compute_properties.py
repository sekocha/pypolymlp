#!/usr/bin/env python
import argparse

import numpy as np

from pypolymlp.calculator.compute_features import (
    compute_from_polymlp_lammps,
    update_types,
)
from pypolymlp.core.interface_vasp import (
    parse_structures_from_poscars,
    parse_structures_from_vaspruns,
)
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.cxx.lib import libmlpcpp
from pypolymlp.mlp_gen.features import structures_to_mlpcpp_obj


def compute_properties_slow(st_dicts, pot=None, params_dict=None, coeffs=None):
    """
    Return
    ------
    energies: unit: eV/supercell (n_str)
    forces: unit: eV/angstrom (n_str, 3, n_atom)
    stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                unit: eV/supercell
    """
    print(
        "Properties calculations for",
        len(st_dicts),
        "structures: Using a memory efficient but slow algorithm",
    )

    if pot is not None:
        params_dict, mlp_dict = load_mlp_lammps(filename=pot)
        params_dict["element_swap"] = False
        coeffs = mlp_dict["coeffs"] / mlp_dict["scales"]

    element_order = params_dict["elements"]
    st_dicts = update_types(st_dicts, element_order)
    axis_array, positions_c_array, types_array, _ = structures_to_mlpcpp_obj(st_dicts)

    obj = libmlpcpp.PotentialProperties(
        params_dict, coeffs, axis_array, positions_c_array, types_array
    )
    """
    PotentialProperties: Return
    ----------------------------
    energies = obj.get_e(), (n_str)
    forces = obj.get_f(), (n_str, n_atom*3) in the order of (n_atom, 3)
    stresses = obj.get_s(), (n_str, 6)
                             in the order of xx, yy, zz, xy, yz, zx
    """
    energies, forces, stresses = obj.get_e(), obj.get_f(), obj.get_s()
    forces = [np.array(f).reshape((-1, 3)).T for f in forces]

    return energies, forces, stresses


def compute_properties(st_dicts, pot=None, params_dict=None, coeffs=None):
    """
    Return
    ------
    energies: unit: eV/supercell (n_str)
    forces: unit: eV/angstrom (n_str, 3, n_atom)
    stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                unit: eV/supercell
    """
    print(
        "Properties calculations for",
        len(st_dicts),
        "structures: Using a fast algorithm",
    )

    if pot is not None:
        params_dict, mlp_dict = load_mlp_lammps(filename=pot)
        params_dict["element_swap"] = False
        coeffs = mlp_dict["coeffs"] / mlp_dict["scales"]

    element_order = params_dict["elements"]
    st_dicts = update_types(st_dicts, element_order)
    axis_array, positions_c_array, types_array, _ = structures_to_mlpcpp_obj(st_dicts)

    obj = libmlpcpp.PotentialPropertiesFast(
        params_dict, coeffs, axis_array, positions_c_array, types_array
    )
    """
    PotentialProperties: Return
    ----------------------------
    energies = obj.get_e(), (n_str)
    forces = obj.get_f(), (n_str, n_atom, 3)
    stresses = obj.get_s(), (n_str, 6)
                             in the order of xx, yy, zz, xy, yz, zx
    """
    energies, forces, stresses = obj.get_e(), obj.get_f(), obj.get_s()
    forces = [np.array(f).T for f in forces]

    return energies, forces, stresses


def compute_energies(st_dicts, pot):

    x, mlp_dict = compute_from_polymlp_lammps(
        st_dicts, pot=pot, force=False, stress=False
    )
    coeffs = mlp_dict["coeffs"] / mlp_dict["scales"]
    return x @ coeffs


def convert_stresses_in_gpa(stresses, st_dicts):

    volumes = np.array([st["volume"] for st in st_dicts])
    stresses_gpa = np.zeros(stresses.shape)
    for i in range(6):
        stresses_gpa[:, i] = stresses[:, i] / volumes * 160.21766208
    return stresses_gpa


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscars",
        nargs="*",
        type=str,
        default=None,
        help="poscar files",
    )
    parser.add_argument(
        "--vaspruns",
        nargs="*",
        type=str,
        default=None,
        help="vasprun files",
    )
    parser.add_argument(
        "--phono3py_yaml", type=str, default=None, help="phono3py.yaml file"
    )
    parser.add_argument(
        "--pot", type=str, default="polymlp.lammps", help="polymlp file"
    )
    args = parser.parse_args()

    if args.poscars is not None:
        structures = parse_structures_from_poscars(args.poscars)
    elif args.vaspruns is not None:
        structures = parse_structures_from_vaspruns(args.vaspruns)
    elif args.phono3py_yaml is not None:
        from pypolymlp.core.interface_phono3py import (
            parse_structures_from_phono3py_yaml,
        )

        structures = parse_structures_from_phono3py_yaml(args.phono3py_yaml)

    """
    energies = compute_energies(args.pot, structures)
    """
    energies, forces, stresses = compute_properties(structures, pot=args.pot)
    stresses_gpa = convert_stresses_in_gpa(stresses, structures)

    np.set_printoptions(suppress=True)
    np.save("polymlp_energies.npy", energies)
    np.save("polymlp_forces.npy", forces)
    np.save("polymlp_stress_tensors.npy", stresses_gpa)

    if len(forces) == 1:
        print(" energy =", energies[0], "(eV/cell)")
        print(" forces =")
        for i, f in enumerate(forces[0].T):
            print("  - atom", i, ":", f)
        stress = stresses_gpa[0]
        print(" stress tensors =")
        print("  - xx, yy, zz:", stress[0:3])
        print("  - xy, yz, zx:", stress[3:6])
        print("---------")
        print(
            " polymlp_energies.npy, polymlp_forces.npy,",
            "and polymlp_stress_tensors.npy are generated.",
        )
    else:
        print(
            " polymlp_energies.npy, polymlp_forces.npy,",
            "and polymlp_stress_tensors.npy are generated.",
        )
