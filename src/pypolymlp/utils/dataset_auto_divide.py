"""Utility functions for dividing datasets according to property values."""

import os
import shutil
from typing import Optional

import numpy as np

from pypolymlp.core.dataset_utils import DatasetDFT
from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.core.utils import split_train_test


def auto_divide_vaspruns(
    vaspruns: list[str],
    path_output: str = "./",
    verbose: bool = False,
):
    """Divide a dataset into training and test datasets automatically."""
    dft = set_dataset_from_vaspruns(vaspruns)

    train1, train2, train0, test1, test2, test0 = _split_three_datasets(dft)
    if verbose:
        print(" - Subset size (train1):", len(train1))
        print(" - Subset size (train2):", len(train2))
        print(" - Subset size (train_high_e):", len(train0))
        print(" - Subset size (test1): ", len(test1))
        print(" - Subset size (test2): ", len(test2))
        print(" - Subset size (test_high_e):", len(test0))

    vaspruns = np.array(vaspruns)
    os.makedirs(path_output, exist_ok=True)

    f = open(path_output + "/polymlp.in.append", "w")
    print(file=f)
    path_output = path_output + "/vaspruns/"
    if len(train1):
        _copy_vaspruns(vaspruns[train1], "train1", path_output=path_output)
        print("train_data vaspruns/train1/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("train_data vaspruns/train1/vaspruns-*.xml True 1.0")
    if len(train2):
        _copy_vaspruns(vaspruns[train2], "train2", path_output=path_output)
        print("train_data vaspruns/train2/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("train_data vaspruns/train2/vaspruns-*.xml True 1.0")
    if len(train0):
        _copy_vaspruns(vaspruns[train0], "train_high_e", path_output=path_output)
        print(
            "train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1",
            file=f,
        )
        if verbose:
            print("train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1")

    if len(test1):
        _copy_vaspruns(vaspruns[test1], "test1", path_output=path_output)
        print("test_data vaspruns/test1/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("test_data vaspruns/test1/vaspruns-*.xml True 1.0")
    if len(test2):
        _copy_vaspruns(vaspruns[test2], "test2", path_output=path_output)
        print("test_data vaspruns/test2/vaspruns-*.xml True 1.0", file=f)
        if verbose:
            print("test_data vaspruns/test2/vaspruns-*.xml True 1.0")
    if len(test0):
        _copy_vaspruns(vaspruns[test0], "test_high_e", path_output=path_output)
        print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1", file=f)
        if verbose:
            print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1")
    f.close()


def _extract_properties_from_dataset(dft: DatasetDFT):
    """Extract energies, stddev of forces, and volumes."""
    e_all = dft.energies / dft.total_n_atoms
    vol_all = dft.volumes / dft.total_n_atoms

    f_std_all = []
    begin = 0
    for n in dft.total_n_atoms:
        end = begin + 3 * n
        f_std_all.append(np.std(dft.forces[begin:end]))
        begin = end
    f_std_all = np.array(f_std_all)
    return (e_all, f_std_all, vol_all)


def _set_threshold_energy(
    e_all: Optional[np.ndarray] = None,
    eth: Optional[float] = None,
    e_ratio: float = 0.25,
):
    """Set energy threshold."""
    if eth is None:
        eth_from_min = np.std(e_all) * e_ratio
        eth = np.min(e_all) + eth_from_min
    return eth


def _set_threshold_force(
    f_std_all: Optional[np.ndarray] = None,
    fth: Optional[float] = None,
    f_ratio: float = 1.0,
):
    """Set force standard deviation threshold."""
    if fth is None:
        fth = np.average(f_std_all) * f_ratio
    return fth


def _set_threshold_volume(
    vol_all: Optional[np.ndarray] = None,
    volth: Optional[float] = None,
    vol_ratio: float = 2.0,
):
    """Set volume threshold."""
    if volth is None:
        volth = np.average(vol_all) * vol_ratio
    return volth


def _split_two_datasets(
    dft: DatasetDFT,
    eth: Optional[float] = None,
    fth: Optional[float] = None,
    volth: Optional[float] = None,
    e_ratio: float = 0.25,
    f_ratio: float = 1.0,
    vol_ratio: float = 2.0,
    train_ratio: float = 0.9,
    verbose: bool = False,
):
    """Split a dataset into two datasets according to properties."""
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(dft)
    eth = _set_threshold_energy(e_all, eth=eth, e_ratio=e_ratio)
    fth = _set_threshold_force(f_std_all, fth=fth, f_ratio=f_ratio)
    volth = _set_threshold_volume(vol_all, volth=volth, vol_ratio=vol_ratio)

    if verbose:
        print(" energy threshold: ", eth)
        print(" force threshold:  ", fth)
        print(" volume threshold: ", volth)

    dataset_bools = (e_all <= eth) & (f_std_all <= fth) & (vol_all <= volth)
    set1 = np.where(dataset_bools)[0]
    set2 = np.where(~dataset_bools)[0]
    train1, test1 = split_train_test(set1, train_ratio=train_ratio)
    train2, test2 = split_train_test(set2, train_ratio=train_ratio)
    return train1, train2, test1, test2


def _split_three_datasets(
    dft: DatasetDFT,
    eth: Optional[float] = None,
    fth: Optional[float] = None,
    volth: Optional[float] = None,
    e_ratio: float = 0.25,
    f_ratio: float = 1.0,
    vol_ratio: float = 2.0,
    verbose: bool = False,
):
    """Split a dataset into three datasets according to properties."""
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(dft)
    eth = _set_threshold_energy(e_all, eth=eth, e_ratio=e_ratio)
    fth = _set_threshold_force(f_std_all, fth=fth, f_ratio=f_ratio)
    volth = _set_threshold_volume(vol_all, volth=volth, vol_ratio=vol_ratio)

    if verbose:
        print(" energy threshold: ", eth)
        print(" force threshold:  ", fth)
        print(" volume threshold: ", volth)

    bools0 = e_all > 0.0
    bools1 = ~bools0 & (e_all <= eth) & (f_std_all <= fth) & (vol_all <= volth)
    bools2 = ~bools0 & ~bools1
    set0 = np.where(bools0)[0]
    set1 = np.where(bools1)[0]
    set2 = np.where(bools2)[0]

    train1, test1 = split_train_test(set1)
    train2, test2 = split_train_test(set2)
    train0, test0 = split_train_test(set0)
    return train1, train2, train0, test1, test2, test0


def _copy_vaspruns(vaspruns: list, tag: str, path_output: str = "./", suffix: str = ""):
    """Copy vasprun.xml files."""
    dir_output = path_output + "/" + tag + "/"
    os.makedirs(dir_output, exist_ok=True)

    with open(dir_output + "dataset_auto_div.yaml", "w") as f:
        print("datasets:", file=f)
        print("- dataset_id:", tag, file=f)
        print("  structures:", file=f)
        for id1, infile in enumerate(vaspruns):
            print("  - id:  ", id1, file=f)
            print("    file:", infile, file=f)

            outfile = "vaspruns-" + str(id1).zfill(5) + ".xml" + suffix
            shutil.copyfile(infile, dir_output + outfile)
