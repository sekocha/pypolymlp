"""Functions for saving and loading yaml files."""

import io
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
    file: Optional[str] = None,
):
    """Save a structure to a yaml file."""
    np.set_printoptions(legacy="1.21")
    if isinstance(file, str):
        fstream = open(file, "w")
    elif isinstance(file, io.IOBase):
        fstream = file
    else:
        raise RuntimeError("file is not str or io.IOBase")

    print(tag + ":", file=fstream)
    print_array2d(cell.axis.T, "axis", fstream, indent_l=2)
    print_array2d(cell.positions.T, "positions", fstream, indent_l=2)
    print("  n_atoms:  ", list(cell.n_atoms), file=fstream)
    print("  types:    ", list(cell.types), file=fstream)
    print("  elements: ", list(cell.elements), file=fstream)
    if cell.volume is not None:
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
    file: str = "cells.yaml",
):
    """Save unitcell and supercell in yaml file."""
    if isinstance(file, str):
        f = open(file, "w")

    save_cell(unitcell, tag="unitcell", file=f)
    save_cell(supercell, tag="supercell", file=f)
    f.close()


def load_cell(
    filename: Optional[str] = None,
    yaml_data: Optional[dict] = None,
    tag: str = "unitcell",
):
    """Parse structure in yaml data."""
    if filename is not None:
        yaml_data = yaml.safe_load(open(filename))
    cell_dict = yaml_data[tag]
    cell_dict["axis"] = np.array(cell_dict["axis"], dtype=float).T
    cell_dict["positions"] = np.array(cell_dict["positions"], dtype=float).T
    cell = PolymlpStructure(**cell_dict)
    return cell


def load_cells(
    filename: Optional[str] = None,
    yaml_data: Optional[dict] = None,
):
    """Parse unitcell and supercell in yaml data."""
    if filename is not None:
        yaml_data = yaml.safe_load(open(filename))
    unitcell = load_cell(yaml_data=yaml_data, tag="unitcell")
    supercell = load_cell(yaml_data=yaml_data, tag="supercell")
    supercell.supercell_matrix = np.array(yaml_data["supercell"]["supercell_matrix"])
    return unitcell, supercell
