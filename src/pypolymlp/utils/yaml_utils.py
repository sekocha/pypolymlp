#!/usr/bin/env python
import numpy as np
import yaml


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
    print("  volume:   ", cell["volume"], file=fstream)

    if tag == "supercell":
        if "n_unitcells" in cell:
            print("  n_unitcells: ", cell["n_unitcells"], file=fstream)
        print_array2d(
            cell["supercell_matrix"],
            "supercell_matrix",
            fstream,
            indent_l=2,
        )

    print("", file=fstream)


def save_cells(unitcell, supercell, filename="cells.yaml"):

    f = open(filename, "w")
    save_cell(unitcell, tag="unitcell", fstream=f)
    save_cell(supercell, tag="supercell", fstream=f)
    f.close()


def load_cells(filename="cells.yaml"):

    yml_data = yaml.safe_load(open(filename))
    unitcell = yml_data["unitcell"]
    unitcell["axis"] = np.array(unitcell["axis"]).T
    unitcell["positions"] = np.array(unitcell["positions"]).T

    supercell = yml_data["supercell"]
    supercell["axis"] = np.array(supercell["axis"]).T
    supercell["positions"] = np.array(supercell["positions"]).T
    supercell["supercell_matrix"] = np.array(supercell["supercell_matrix"])

    return unitcell, supercell
