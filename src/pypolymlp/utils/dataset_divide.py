"""Functions for dividing datasets according to property values."""

import os
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies
from pypolymlp.utils.dataset_divide_utils import (
    copy_vaspruns,
    split_datasets,
    split_datasets_alloy,
)


def _parse_vaspruns(
    vaspruns: list[str],
    elements: tuple,
    functional: Literal["PBE", "PBEsol"] = "PBE",
):
    """Parse vaspruns for dataset division."""
    if vaspruns is None:
        raise RuntimeError("Vasprun files not found.")
    if elements is None:
        raise RuntimeError("Element strings not found.")

    dft = set_dataset_from_vaspruns(vaspruns, elements=elements)
    try:
        atom_e = get_atomic_energies(elements, functional=functional)[0]
        dft.apply_atomic_energy(atom_e)
    except:
        print("Atomic energies not found.")
        atom_e = None
        pass
    return dft, atom_e


def _get_dataset_attrs(n_divide: int = 3):
    """Weight and include_force values."""
    if n_divide == 1:
        weights = (1.0,)
        forces = (True,)
    elif n_divide == 2:
        weights = (0.1, 1.0)
        forces = (True, True)
    else:
        weights = tuple([0.1] + [1.0] * (n_divide - 1))
        forces = tuple([True, True] + [True] * (n_divide - 1))
    return np.array(weights), np.array(forces)


def auto_divide_vaspruns(
    vaspruns: list[str],
    elements: tuple,
    n_divide: int = 3,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    path_output: Optional[str] = None,
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    dft, atom_e = _parse_vaspruns(vaspruns, elements, functional=functional)
    if len(elements) == 1:
        data_attrs = split_datasets(dft, n_divide=n_divide, verbose=verbose)
    else:
        data_attrs = split_datasets_alloy(
            dft,
            elements,
            n_divide=n_divide,
            verbose=verbose,
        )

    if verbose:
        print("Dataset sizes:", flush=True)
        for i, attr in enumerate(data_attrs):
            print("- Group" + str(i + 1) + ":", flush=True)
            print("  train: ", len(attr.train), flush=True)
            print("  test:  ", len(attr.test), flush=True)

    vaspruns = np.array(vaspruns)
    weights, forces = _get_dataset_attrs(n_divide)
    if len(elements) > 1:
        n_tiles = len(data_attrs) // n_divide
        weights = np.tile(weights, n_tiles)
        forces = np.tile(forces, n_tiles)

    path = "./polymlp_datasets/" if path_output is None else path_output + "/"
    os.makedirs(path, exist_ok=True)
    f = open(path + "/polymlp.in.append", "w")

    if atom_e is not None:
        print("n_type", len(atom_e), file=f)
        print("atomic_energy ", end="", file=f)
        for e in atom_e:
            print(e, end=" ", file=f)
        print(file=f)

    if verbose:
        print("-----", flush=True)
    for i, attr in enumerate(data_attrs):
        weight = weights[i]
        force = forces[i]
        if len(attr.train) > 0:
            tag = "train" + str(i + 1)
            files = path + tag + "/*.xml"
            copy_vaspruns(vaspruns[attr.train], tag, path_output=path)
            print("train_data", files, str(force), str(weight), file=f)
            if verbose:
                print("train_data", files, str(force), str(weight), flush=True)

        if len(attr.test) > 0:
            tag = "test" + str(i + 1)
            files = path + tag + "/*.xml"
            copy_vaspruns(vaspruns[attr.test], tag, path_output=path)
            print("test_data", files, str(force), str(weight), file=f)
            if verbose:
                print("test_data", files, str(force), str(weight), flush=True)
    f.close()
