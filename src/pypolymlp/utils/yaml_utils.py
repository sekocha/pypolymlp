"""Functions for saving and loading yaml files."""

from typing import Optional

import numpy as np
import yaml

from pypolymlp.core.data_format import PolymlpStructure


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


def save_cell(
    cell: PolymlpStructure,
    tag: str = "unitcell",
    fstream=None,
    filename: Optional[str] = None,
):
    """Save a cell in yaml file."""
    if fstream is None:
        fstream = open(filename, "w")

    print(tag + ":", file=fstream)
    print_array2d(cell.axis.T, "axis", fstream, indent_l=2)
    print_array2d(cell.positions.T, "positions", fstream, indent_l=2)
    print("  n_atoms:  ", list(cell.n_atoms), file=fstream)
    print("  types:    ", list(cell.types), file=fstream)
    print("  elements: ", list(cell.elements), file=fstream)
    print("  volume:   ", cell.volume, file=fstream)

    if tag == "supercell":
        if cell.n_unitcells is not None:
            print("  n_unitcells: ", cell.n_unitcells, file=fstream)
        print_array2d(
            cell.supercell_matrix,
            "supercell_matrix",
            fstream,
            indent_l=2,
        )

    print("", file=fstream)


def save_cells(
    unitcell: PolymlpStructure,
    supercell: PolymlpStructure,
    filename: str = "cells.yaml",
):
    """Save unitcell and supercell in yaml file."""
    f = open(filename, "w")
    save_cell(unitcell, tag="unitcell", fstream=f)
    save_cell(supercell, tag="supercell", fstream=f)
    f.close()


def load_cells(filename="cells.yaml"):
    """Parse unitcell and supercell in yaml data."""
    yml_data = yaml.safe_load(open(filename))
    cell_dict = yml_data["unitcell"]
    cell_dict["axis"] = np.array(cell_dict["axis"]).T
    cell_dict["positions"] = np.array(cell_dict["positions"]).T
    unitcell = PolymlpStructure(**cell_dict)

    cell_dict = yml_data["supercell"]
    cell_dict["axis"] = np.array(cell_dict["axis"]).T
    cell_dict["positions"] = np.array(cell_dict["positions"]).T
    cell_dict["supercell_matrix"] = np.array(cell_dict["supercell_matrix"])
    supercell = PolymlpStructure(**cell_dict)
    return unitcell, supercell
