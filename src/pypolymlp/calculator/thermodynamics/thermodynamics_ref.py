"""Utility functions for initializing thermodynamic property calculation."""

import copy
import os
from typing import Optional

import numpy as np
import yaml

from pypolymlp.calculator.compute_phonon import calculate_harmonic_properties_from_fc2
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_grid import (
    GridPointData,
    GridVT,
)
from pypolymlp.core.units import EVtoJmol, EVtoKJmol


def set_reference_paths(
    grid_ti: GridVT,
    ref_fc2: Optional[list] = None,
    decimals: int = 3,
):
    """Set paths of reference FC2 states."""
    path_fc2_dict = dict()
    if ref_fc2 is not None:
        for fc2hdf5 in ref_fc2:
            cwd = "/".join(fc2hdf5.split("/")[:-1])
            yml = yaml.safe_load(open(cwd + "/sscha_results.yaml"))
            vol = yml["unitcell"]["volume"]
            n_atom = len(yml["unitcell"]["elements"])
            vol = np.round(vol / n_atom, decimals)
            path_fc2_dict[vol] = fc2hdf5

    for i, j, d in grid_ti:
        if d.is_empty:
            continue
        vol = np.round(d.volume, decimals)
        cwd = "/".join(d.path_yaml.split("/")[:-1])
        fc2hdf5 = cwd + "/fc2.hdf5"
        if os.path.exists(fc2hdf5):
            d.path_fc2 = fc2hdf5
        elif vol in path_fc2_dict:
            d.path_fc2 = path_fc2_dict[vol]
    return grid_ti


def copy_reference_states(grid_sscha: GridVT, grid_ti: GridVT):
    """Copy reference states"""
    if grid_sscha.data.shape != grid_ti.data.shape:
        raise RuntimeError("Shapes mismatch.")

    for i, j, d in grid_ti:
        if d.is_empty:
            continue
        d.restart = grid_sscha[i, j].restart
        d.static_potential = d.restart.static_potential
    return grid_ti


def _calculate_harmonic_properties(
    res: Restart,
    path_fc2: str,
    mesh: tuple = (10, 10, 10),
    temperatures: Optional[np.ndarray] = None,
):
    """Calculate harmonic thermodynamic properties."""
    if temperatures is None:
        temperatures = [res.temperature]

    tp_dict = calculate_harmonic_properties_from_fc2(
        res.unitcell,
        res.supercell_matrix,
        path_fc2=path_fc2,
        mesh=mesh,
        temperatures=temperatures,
    )
    return tp_dict


def calculate_reference_grid(
    grid_ti: GridVT,
    mesh: tuple = (10, 10, 10),
    decimals: int = 3,
):
    """Return reference properties.

    Harmonic phonon properties calculated with SSCHA FC2 and SSCHA frequencies
    at the lowest temperature are used as reference free energy, reference entropy,
    and reference heat capacity to fit properties with respect to temperature.
    """
    grid_ref = copy.deepcopy(grid_ti)
    for i, j, d in grid_ti:
        if d.is_empty:
            continue

        # paths = [d2.path_fc2 for d2 in d1]
        # if paths[0] != "fc2.hdf5" and np.all(paths == paths[0]):
        # else:
        tp_dict = _calculate_harmonic_properties(
            d.restart,
            d.path_fc2,
            temperatures=d.temperature,
        )
        n_atom = len(d.restart.unitcell.elements)
        free_energy = tp_dict["free_energy"][0] / EVtoKJmol / n_atom
        free_energy += d.static_potential
        entropy = tp_dict["entropy"][0] / EVtoJmol / n_atom
        heat_capacity = tp_dict["heat_capacity"][0] / n_atom

        grid_point = GridPointData(
            volume=d.volume,
            temperature=d.temperature,
            free_energy=free_energy,
            entropy=entropy,
            heat_capacity=heat_capacity,
        )
        grid_ref[i, j] = grid_point

    return grid_ref
