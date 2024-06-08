#!/usr/bin/env python
import os

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5

from pypolymlp.core.utils import kjmol_to_ev
from pypolymlp.utils.phonopy_utils import phonopy_supercell


def temperature_setting(args):

    if args.temp is not None:
        if np.isclose(args.temp, round(args.temp)):
            args.temp = int(args.temp)
        temp_array = [args.temp]
    else:
        if np.isclose(args.temp_min, round(args.temp_min)):
            args.temp_min = int(args.temp_min)
        if np.isclose(args.temp_max, round(args.temp_max)):
            args.temp_max = int(args.temp_max)
        if np.isclose(args.temp_step, round(args.temp_step)):
            args.temp_step = int(args.temp_step)
        temp_array = np.arange(args.temp_min, args.temp_max + 1, args.temp_step)
        if args.ascending_temp is False:
            temp_array = temp_array[::-1]
    args.temperatures = temp_array
    return args


def n_steps_setting(args, n_atom_supercell=None):

    if args.n_samples is None:
        n_steps_unit = round(3200 / n_atom_supercell)
        args.n_steps = 20 * n_steps_unit
        args.n_steps_final = 100 * n_steps_unit
    else:
        args.n_steps, args.n_steps_final = args.n_samples
    return args


def print_structure(cell):

    print(" # structure ")
    print("  - elements:     ", cell["elements"])
    print("  - axis:         ", cell["axis"].T[0])
    print("                  ", cell["axis"].T[1])
    print("                  ", cell["axis"].T[2])
    print("  - positions:    ", cell["positions"].T[0])
    if cell["positions"].shape[1] > 1:
        for pos in cell["positions"].T[1:]:
            print("                  ", pos)


def print_parameters(supercell, args):

    print(" # parameters")
    print("  - supercell:    ", supercell[0])
    print("                  ", supercell[1])
    print("                  ", supercell[2])
    print("  - temperatures: ", args.temperatures[0])
    if len(args.temperatures) > 1:
        for t in args.temperatures[1:]:
            print("                  ", t)

    if isinstance(args.pot, list):
        for p in args.pot:
            print("  - Polynomial ML potential:  ", os.path.abspath(p))
    else:
        print("  - Polynomial ML potential:  ", os.path.abspath(args.pot))

    print("  - FC tolerance:             ", args.tol)
    print("  - max iter:                 ", args.max_iter)
    print("  - num samples:              ", args.n_steps)
    print("  - num samples (last iter.): ", args.n_steps_final)
    print("  - q-mesh:                   ", args.mesh)


def print_array1d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream)
    for i, d in enumerate(array):
        print(prefix + " -", d, file=fstream)


def print_array2d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream)
    for i, d in enumerate(array):
        print(prefix + " -", list(d), file=fstream)


def save_cell(cell, tag="unitcell", fstream=None, filename=None):

    if fstream is None:
        fstream = open(filename, "w")

    print(tag + ":", file=fstream)
    print_array2d(cell["axis"].T, "axis", fstream, indent_l=2)
    print_array2d(cell["positions"].T, "positions", fstream, indent_l=2)
    print("  n_atoms:  ", list(cell["n_atoms"]), file=fstream)
    print("  types:    ", list(cell["types"]), file=fstream)
    print("  elements: ", list(cell["elements"]), file=fstream)

    if tag == "supercell":
        print("  n_unitcells: ", cell["n_unitcells"], file=fstream)
        print_array2d(
            cell["supercell_matrix"],
            "supercell_matrix",
            fstream,
            indent_l=2,
        )

    print("", file=fstream)


def save_sscha_yaml(sscha, args, filename="sscha_results.yaml"):

    sscha_dict = sscha.properties
    log_dict = sscha.logs
    unitcell = sscha.unitcell_dict
    supercell_matrix = sscha.supercell_matrix

    f = open(filename, "w")
    print("parameters:", file=f)
    if isinstance(args.pot, list):
        for p in args.pot:
            print("  pot:     ", os.path.abspath(p), file=f)
    else:
        print("  pot:     ", os.path.abspath(args.pot), file=f)
    print("  temperature:   ", sscha_dict["temperature"], file=f)
    print("  n_steps:       ", args.n_steps, file=f)
    print("  n_steps_final: ", args.n_steps_final, file=f)
    print("  tolerance:     ", args.tol, file=f)
    print("  mixing:        ", args.mixing, file=f)
    print("  mesh_phonon:   ", list(args.mesh), file=f)
    print("", file=f)

    print("properties:", file=f)
    print("  free_energy:     ", sscha_dict["free_energy"], file=f)
    print("  static_potential:", sscha_dict["static_potential"], file=f)

    print("status:", file=f)
    print("  delta_fc: ", log_dict["delta"], file=f)
    print("  converge: ", log_dict["converge"], file=f)
    print("", file=f)

    save_cell(unitcell, tag="unitcell", fstream=f)
    print("supercell_matrix:", file=f)
    print(" -", list(supercell_matrix[0]), file=f)
    print(" -", list(supercell_matrix[1]), file=f)
    print(" -", list(supercell_matrix[2]), file=f)
    print("", file=f)

    print("logs:", file=f)
    logs = log_dict["history"]
    print_array1d(logs["free_energy"], "free_energy", f, indent_l=2)
    print("", file=f)
    print_array1d(logs["harmonic_potential"], "harmonic_potential", f, indent_l=2)
    print("", file=f)
    print_array1d(logs["average_potential"], "average_potential", f, indent_l=2)
    print("", file=f)
    print_array1d(
        logs["anharmonic_free_energy"],
        "anharmonic_free_energy",
        f,
        indent_l=2,
    )
    print("", file=f)

    f.close()


class Restart:

    def __init__(self, res_yaml, fc2hdf5=None, unit="kJ/mol"):

        self.yaml = res_yaml
        self.__unit = unit

        self.load_sscha_yaml()
        if fc2hdf5 is not None:
            self.__fc2 = read_fc2_from_hdf5(fc2hdf5)

    def load_sscha_yaml(self):

        yaml_data = yaml.safe_load(open(self.yaml))

        self.__pot = yaml_data["parameters"]["pot"]
        self.__temp = yaml_data["parameters"]["temperature"]

        self.__free_energy = yaml_data["properties"]["free_energy"]
        self.__static_potential = yaml_data["properties"]["static_potential"]

        self.__delta_fc = yaml_data["status"]["delta_fc"]
        self.__converge = yaml_data["status"]["converge"]
        self.__logs = yaml_data["logs"]

        self.__unitcell = yaml_data["unitcell"]
        self.__unitcell["axis"] = np.array(self.__unitcell["axis"]).T
        self.__unitcell["positions"] = np.array(self.__unitcell["positions"]).T

        self.__supercell_matrix = np.array(yaml_data["supercell_matrix"])
        self.__n_atom_unitcell = len(self.__unitcell["elements"])

    @property
    def polymlp(self):
        return self.__pot

    @property
    def temperature(self):
        return self.__temp

    @property
    def free_energy(self):
        if self.__unit == "kJ/mol":
            return self.__free_energy
        elif self.__unit == "eV/atom":
            return kjmol_to_ev(self.__free_energy) / self.__n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")

    @property
    def static_potential(self):
        if self.__unit == "kJ/mol":
            return self.__static_potential
        elif self.__unit == "eV/atom":
            return kjmol_to_ev(self.__static_potential) / self.__n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")

    @property
    def logs(self):
        return self.__logs

    @property
    def force_constants(self):
        return self.__fc2

    @property
    def unitcell(self):
        return self.__unitcell

    @property
    def supercell_matrix(self):
        return self.__supercell_matrix

    @property
    def n_unitcells(self):
        return int(round(np.linalg.det(self.__supercell_matrix)))

    @property
    def supercell(self):
        cell = phonopy_supercell(
            self.__unitcell,
            supercell_matrix=self.__supercell_matrix,
            return_phonopy=False,
        )
        return cell

    @property
    def supercell_phonopy(self):
        cell = phonopy_supercell(
            self.__unitcell,
            supercell_matrix=self.__supercell_matrix,
            return_phonopy=True,
        )
        return cell

    @property
    def unitcell_volume(self):
        volume = np.linalg.det(self.__unitcell["axis"])
        if self.__unit == "kJ/mol":
            return volume
        elif self.__unit == "eV/atom":
            return volume / self.__n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")
