#!/usr/bin/env python
from collections import defaultdict

import numpy as np

from pypolymlp.core.utils import permute_atoms


def convert_disps_to_positions(disps, axis, positions):
    """disps: (n_str, 3, n_atoms) # Angstrom"""
    axis_inv = np.linalg.inv(axis)
    np.set_printoptions(suppress=True)
    positions_all = np.array([positions + (axis_inv @ d) for d in disps])
    return positions_all


def set_dft_dict(forces, energies, positions_all, st_dict, element_order=None):
    """
    Parameters
    ----------
    forces: (n_str, 3, n_atom)
    energies: (n_str)
    positions_all: (n_str, 3, n_atom)
    st_dict: structure without displacements

    Return
    ------
    dft_dict: DFT training or test dataset in pypolymlp format
    """
    dft_dict = defaultdict(list)
    dft_dict["energy"] = energies
    dft_dict["stress"] = np.zeros(forces.shape[0] * 6)
    for positions_iter, forces_iter in zip(positions_all, forces):
        st = dict()
        st["axis"] = st_dict["axis"]
        st["positions"] = positions_iter
        st["n_atoms"] = st_dict["n_atoms"]
        st["elements"] = st_dict["elements"]
        st["types"] = st_dict["types"]
        st["volume"] = st_dict["volume"]

        if element_order is not None:
            st, forces_iter = permute_atoms(st, forces_iter, element_order)

        dft_dict["force"].extend(forces_iter.T.reshape(-1))
        dft_dict["structures"].append(st)
    dft_dict["force"] = np.array(dft_dict["force"])

    if element_order is not None:
        dft_dict["elements"] = element_order
    else:
        elements_rep = dft_dict["structures"][0]["elements"]
        dft_dict["elements"] = sorted(set(elements_rep), key=elements_rep.index)

    dft_dict["total_n_atoms"] = np.array(
        [sum(st["n_atoms"]) for st in dft_dict["structures"]]
    )
    n_data = len(dft_dict["structures"])
    dft_dict["filenames"] = ["disp-" + str(i + 1).zfill(5) for i in range(n_data)]
    return dft_dict


def get_structures_from_multiple_positions(st_dict, positions_all):
    """positions_all: (n_str, 3, n_atom)"""
    st_dicts = []
    for positions_iter in positions_all:
        st = dict()
        st["axis"] = st_dict["axis"]
        st["positions"] = positions_iter
        st["n_atoms"] = st_dict["n_atoms"]
        st["elements"] = st_dict["elements"]
        st["types"] = st_dict["types"]
        st["volume"] = st_dict["volume"]
        st_dicts.append(st)
    return st_dicts


def get_structures_from_displacements(disps, st_dict):
    """disps: (n_str, 3, n_atoms)"""
    positions_all = convert_disps_to_positions(
        disps, st_dict["axis"], st_dict["positions"]
    )
    st_dicts = get_structures_from_multiple_positions(st_dict, positions_all)
    return st_dicts


def generate_random_const_displacements(
    st_dict, n_samples=100, displacements=0.03, is_plusminus=False
):

    positions = st_dict["positions"]
    disps = []
    for i in range(n_samples):
        # rand = np.random.rand(3, positions.shape[1]) - 0.5
        rand = np.random.randn(3, positions.shape[1])
        rand = rand / np.linalg.norm(rand, axis=0)
        disps.append(rand * displacements)
        if is_plusminus:
            disps.append(-rand * displacements)
    disps = np.array(disps)

    st_dicts = get_structures_from_displacements(disps, st_dict)
    return disps, st_dicts


def generate_random_displacements(st_dict, n_samples=100, displacements=0.03):

    positions = st_dict["positions"]
    disps = np.random.rand(n_samples, 3, positions.shape[1]) - 0.5
    disps = (2.0 * displacements) * disps

    st_dicts = get_structures_from_displacements(disps, st_dict)
    return disps, st_dicts
