"""Utility functions for dividing datasets according to property values."""

import os
import shutil
from typing import Optional

import numpy as np

from pypolymlp.core.dataset_utils import DatasetDFT
from pypolymlp.core.utils import split_train_test


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


def split_two_datasets(
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


def split_three_datasets(
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


def split_datasets(dft: DatasetDFT, verbose: bool = False):
    """Split a dataset into three datasets according to properties."""
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(dft)

    e_min = np.min(e_all)
    f_std_average = np.average(f_std_all)

    group_e = np.zeros(e_all.shape[0], dtype=int)
    for i, e_ratio in enumerate((0, 0.3, 0.6, 0.8)):
        eth = e_min * e_ratio
        group_e[e_all <= eth] = i + 1

    group_f = np.zeros(e_all.shape[0], dtype=int)
    for i, f_ratio in enumerate((4, 2, 1, 0.5)):
        fth = f_std_average * f_ratio
        group_f[f_std_all <= fth] = i + 1

    group = np.maximum(group_e, group_f)
    print("---")
    print(np.count_nonzero(group == 4))
    print(np.count_nonzero(group == 3))
    print(np.count_nonzero(group == 2))
    print(np.count_nonzero(group == 1))
    print(np.count_nonzero(group == 0))

    datasets = []
    for group_id in range(5):
        set1 = np.where(group == group_id)[0]
        train, test = split_train_test(set1)
        datasets.append((train, test))
    return datasets


def copy_vaspruns(vaspruns: list, tag: str, path_output: str = "./", suffix: str = ""):
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
