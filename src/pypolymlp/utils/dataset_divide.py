"""Functions for dividing datasets according to property values."""

import os
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies
from pypolymlp.utils.dataset_divide_utils import copy_vaspruns, split_datasets


def auto_divide_vaspruns(
    vaspruns: list[str],
    elements: tuple,
    n_divide: int = 3,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    path_output: Optional[str] = None,
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    if vaspruns is None:
        raise RuntimeError("Vasprun files not found.")
    if elements is None:
        raise RuntimeError("Element strings not found.")

    dft = set_dataset_from_vaspruns(vaspruns, element_order=elements)
    try:
        atom_e = get_atomic_energies(elements, functional=functional)[0]
        dft.apply_atomic_energy(atom_e)
    except:
        print("Atomic energies not found.")
        atom_e = None
        pass

    datasets = split_datasets(dft, n_divide=n_divide, verbose=verbose)
    if verbose:
        print("Dataset sizes:", flush=True)
        for i, (train, test) in enumerate(datasets):
            print("- Group" + str(i + 1) + ":", flush=True)
            print("  train: ", len(train), flush=True)
            print("  test:  ", len(test), flush=True)

    vaspruns = np.array(vaspruns)
    if path_output is None:
        path = "./polymlp_datasets/"
    else:
        path = path_output + "/"

    os.makedirs(path, exist_ok=True)
    f = open(path + "/polymlp.in.append", "w")

    if atom_e is not None:
        print("n_type", len(atom_e), file=f)
        print("atomic_enegy ", end="", file=f)
        for e in atom_e:
            print(e, end=" ", file=f)
        print(file=f)

    if n_divide == 1:
        weights = (1.0,)
    elif n_divide == 2:
        weights = (0.1, 1.0)
    else:
        weights = tuple([0.01, 0.1] + [1.0] * (n_divide - 2))

    for i, (train, test) in enumerate(datasets):
        weight = weights[i]
        if len(train) > 0:
            tag = "train" + str(i + 1)
            files = path + tag + "/*.xml"
            copy_vaspruns(vaspruns[train], tag, path_output=path)
            print("train_data", files, "True", str(weight), file=f)
            if verbose:
                print("train_data", files, "True", str(weight), flush=True)

        if len(test) > 0:
            tag = "test" + str(i + 1)
            files = path + tag + "/*.xml"
            copy_vaspruns(vaspruns[test], tag, path_output=path)
            print("test_data", files, "True", str(weight), file=f)
            if verbose:
                print("test_data", files, "True", str(weight), flush=True)
    f.close()


# TODO:
def auto_divide_vaspruns_repository_alloy(
    vaspruns: list[str],
    elements: tuple,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    path_output: Optional[str] = None,
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    dft = set_dataset_from_vaspruns(vaspruns, element_order=elements)
    try:
        atom_e = get_atomic_energies(elements, functional=functional)[0]
        dft.apply_atomic_energy(atom_e)
    except:
        print("Atomic energies not found.")
        atom_e = None
        pass
