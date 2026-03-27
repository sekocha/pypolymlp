"""Utility functions for dividing datasets according to property values."""

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
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


def _extract_properties_from_dataset_alloy(dft: DatasetDFT, elements: tuple):
    """Extract energies, stddev of forces, and volumes."""
    e_all, f_std_all, _ = _extract_properties_from_dataset(dft)

    comp_ids = defaultdict(list)
    for i, st in enumerate(dft.structures):
        comp = st.composition(elements)
        key = tuple(np.round(comp, 5))
        comp_ids[key].append(i)

    return (e_all, f_std_all, comp_ids)


def _split(
    e_all: np.ndarray,
    f_std_all: np.ndarray,
    n_divide: int = 3,
    verbose: bool = False,
):
    """Split dataset."""
    e_min = np.min(e_all)
    f_std_average = np.average(f_std_all)

    e_ratios = np.linspace(0, 1, n_divide)[:-1]
    f_ratios = 2 ** np.linspace(2, -4, n_divide)[:-1]

    if verbose and n_divide > 1:
        print("Energy thresholds:", flush=True)
        print("- Group 1: E >= 0", flush=True)
        for i, (e_ratio, f_ratio) in enumerate(zip(e_ratios, f_ratios)):
            print("- Group " + str(i + 2) + ": E <", e_min * e_ratio, flush=True)

    group_e = np.zeros(e_all.shape[0], dtype=int)
    group_f = np.zeros(e_all.shape[0], dtype=int)
    for i, (e_ratio, f_ratio) in enumerate(zip(e_ratios, f_ratios)):
        eth = e_min * e_ratio
        fth = f_std_average * f_ratio
        group_e[e_all <= eth] = i + 1
        group_f[f_std_all <= fth] = i + 1

    groups = np.maximum(group_e, group_f)
    return groups


@dataclass
class DatasetAttr:
    """Dataclass for dataset attributes."""

    train: np.ndarray
    test: np.ndarray
    composition: Optional[tuple] = None
    energy_ub: Optional[float] = None
    energy_lb: Optional[float] = None
    force_ub: Optional[float] = None
    force_lb: Optional[float] = None


def split_datasets(dft: DatasetDFT, n_divide: int = 6, verbose: bool = False):
    """Split a dataset into three datasets according to properties."""
    if n_divide < 2:
        ids = np.arange(dft.energies.shape[0], dtype=int)
        train, test = split_train_test(ids)
        return [(train, test)]

    e_all, f_std_all, _ = _extract_properties_from_dataset(dft)
    groups = _split(e_all, f_std_all, n_divide=n_divide, verbose=verbose)

    datasets = []
    for group_id in range(n_divide):
        set1 = np.where(groups == group_id)[0]
        train, test = split_train_test(set1)
        attr = DatasetAttr(
            train=train,
            test=test,
        )
        datasets.append(attr)
    return datasets


def split_datasets_alloy(
    dft: DatasetDFT,
    elements: tuple,
    n_divide: int = 6,
    verbose: bool = False,
):
    """Split a dataset into three datasets according to properties."""
    e_all, f_std_all, comp_ids = _extract_properties_from_dataset_alloy(dft, elements)

    end_comp = dict()
    seq = n_divide
    for comp in comp_ids.keys():
        a = np.array(comp)
        if np.any(np.isclose(a, 0.0, atol=1e-8) | np.isclose(a, 1.0, atol=1e-8)):
            if comp not in end_comp:
                end_comp[comp] = seq
                seq += n_divide

    groups = np.zeros(e_all.shape[0], dtype=int)
    for comp, ids in comp_ids.items():
        ids = np.array(ids)
        groups_local = _split(
            e_all[ids],
            f_std_all[ids],
            n_divide=n_divide,
            verbose=verbose,
        )
        if comp in end_comp:
            groups_local += end_comp[comp]
        groups[ids] = groups_local

    datasets = []
    for group_id in range(np.max(groups) + 1):
        set1 = np.where(groups == group_id)[0]
        train, test = split_train_test(set1)

        attr = DatasetAttr(
            train=train,
            test=test,
        )
        datasets.append(attr)
    return datasets


def copy_vaspruns(vaspruns: list, tag: str, path_output: str = "./", suffix: str = ""):
    """Copy vasprun.xml files."""
    path = path_output + "/" + tag + "/"
    os.makedirs(path, exist_ok=True)

    with open(path + "polymlp_autodiv.yaml", "w") as f:
        print("datasets:", file=f)
        print("- dataset_id:", tag, file=f)
        print("  structures:", file=f)
        for id1, infile in enumerate(vaspruns):
            print("  - id:  ", id1, file=f)
            print("    file:", infile, file=f)

            outfile = "vaspruns-" + str(id1).zfill(5) + ".xml" + suffix
            shutil.copyfile(infile, path + outfile)


def _set_threshold_energy(
    e_all: Optional[np.ndarray] = None,
    eth: Optional[float] = None,
    e_ratio: float = 0.25,
):
    """Set energy threshold.

    Deprecated.
    """
    if eth is None:
        eth_from_min = np.std(e_all) * e_ratio
        eth = np.min(e_all) + eth_from_min
    return eth


def _set_threshold_force(
    f_std_all: Optional[np.ndarray] = None,
    fth: Optional[float] = None,
    f_ratio: float = 1.0,
):
    """Set force standard deviation threshold.

    Deprecated.
    """
    if fth is None:
        fth = np.average(f_std_all) * f_ratio
    return fth


def _set_threshold_volume(
    vol_all: Optional[np.ndarray] = None,
    volth: Optional[float] = None,
    vol_ratio: float = 2.0,
):
    """Set volume threshold.

    Deprecated.
    """
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
    """Split a dataset into two datasets according to properties.

    Deprecated.
    """
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
    """Split a dataset into three datasets according to properties.

    Deprecated.
    """
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
