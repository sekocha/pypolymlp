"""Functions for dividing datasets according to property values."""

import os

import numpy as np

from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.utils.dataset_divide_utils import copy_vaspruns, split_three_datasets


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
    path_output: str = "./",
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    dft = set_dataset_from_vaspruns(vaspruns, element_order=elements)
    # atom_e = get_atomic_energies(elements)
    # dft.apply_atomic_energy(atom_e)

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
