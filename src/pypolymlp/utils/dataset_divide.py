"""Functions for dividing datasets according to property values."""

import os
from typing import Literal

import numpy as np

from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies
from pypolymlp.utils.dataset_divide_utils import (
    copy_vaspruns,
    split_datasets,
    split_three_datasets,
)


def auto_divide_vaspruns(
    vaspruns: list[str],
    path_output: str = "./",
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    dft = set_dataset_from_vaspruns(vaspruns)

    train1, train2, train0, test1, test2, test0 = split_three_datasets(dft)
    if verbose:
        print(" - Subset size (train1):      ", len(train1))
        print(" - Subset size (train2):      ", len(train2))
        print(" - Subset size (train_high_e):", len(train0))
        print(" - Subset size (test1):       ", len(test1))
        print(" - Subset size (test2):       ", len(test2))
        print(" - Subset size (test_high_e): ", len(test0))

    vaspruns = np.array(vaspruns)
    os.makedirs(path_output, exist_ok=True)

    f = open(path_output + "/polymlp.in.append", "w")
    print(file=f)
    path_output = path_output + "/vaspruns/"
    if len(train1) > 0:
        copy_vaspruns(vaspruns[train1], "train1", path_output=path_output)
        print("train_data vaspruns/train1/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("train_data vaspruns/train1/vaspruns-*.xml True 1.0")
    if len(train2) > 0:
        copy_vaspruns(vaspruns[train2], "train2", path_output=path_output)
        print("train_data vaspruns/train2/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("train_data vaspruns/train2/vaspruns-*.xml True 1.0")
    if len(train0) > 0:
        copy_vaspruns(vaspruns[train0], "train_high_e", path_output=path_output)
        print(
            "train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1",
            file=f,
        )
        if verbose:
            print("train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1")

    if len(test1) > 0:
        copy_vaspruns(vaspruns[test1], "test1", path_output=path_output)
        print("test_data vaspruns/test1/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("test_data vaspruns/test1/vaspruns-*.xml True 1.0")
    if len(test2) > 0:
        copy_vaspruns(vaspruns[test2], "test2", path_output=path_output)
        print("test_data vaspruns/test2/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("test_data vaspruns/test2/vaspruns-*.xml True 1.0")
    if len(test0) > 0:
        copy_vaspruns(vaspruns[test0], "test_high_e", path_output=path_output)
        print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1", file=f)
        if verbose:
            print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1")
    f.close()


def auto_divide_vaspruns_repository(
    vaspruns: list[str],
    elements: tuple,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    path_output: str = "./",
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

    datasets = split_datasets(dft, verbose=verbose)
    if verbose:
        for i, (train, test) in enumerate(datasets):
            print("- Subset size (train " + str(i + 1) + "):", len(train))
            print("- Subset size (test " + str(i + 1) + "): ", len(test))

    vaspruns = np.array(vaspruns)
    path = path_output + "/vaspruns/"
    os.makedirs(path, exist_ok=True)
    f = open(path + "/polymlp.in.append", "w")

    if atom_e is not None:
        print("atomic_enegy ", end="", file=f)
        for e in atom_e:
            print(e, end=" ", file=f)
        print(file=f)

    for i, (train, test) in enumerate(datasets):
        weight = 0.1 if i == 0 else 1.0
        if len(train) > 0:
            tag = "train" + str(i + 1)
            copy_vaspruns(vaspruns[train], tag, path_output=path)
            print("train_data vaspruns/" + tag + "/*.xml True " + str(weight), file=f)
            if verbose:
                print("train_data vaspruns/" + tag + "/*.xml True " + str(weight))

        if len(test) > 0:
            tag = "test" + str(i + 1)
            copy_vaspruns(vaspruns[test], tag, path_output=path)
            print("test_data vaspruns/" + tag + "/*.xml True " + str(weight), file=f)
            if verbose:
                print("test_data vaspruns/" + tag + "/*.xml True " + str(weight))

    f.close()
