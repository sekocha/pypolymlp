"""Utility functions for initializing thermodynamic property calculation."""

from typing import Optional

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5
from phonopy import Phonopy

from pypolymlp.calculator.md.md_utils import load_thermodynamic_integration_yaml
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import GridPointData
from pypolymlp.core.units import EVtoJmol, EVtoKJmol
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell


def load_sscha_yamls(filenames: tuple[str]) -> list[GridPointData]:
    """Load sscha_results.yaml files."""
    data = []
    for yamlfile in filenames:
        res = Restart(yamlfile, unit="eV/atom")
        n_atom = len(res.unitcell.elements)
        volume = np.round(res.volume, decimals=12) / n_atom
        temp = np.round(res.temperature, decimals=3)
        grid = GridPointData(
            volume=volume,
            temperature=temp,
            data_type="sscha",
            restart=res,
            path_yaml=yamlfile,
            path_fc2="/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5",
        )

        if res.converge and not res.imaginary:
            grid.free_energy = res.free_energy + res.static_potential
            grid.entropy = res.entropy
        else:
            grid.free_energy = None
            grid.entropy = None
        data.append(grid)

    return data


def load_electron_yamls(filenames: tuple[str]) -> list[GridPointData]:
    """Load electron.yaml files."""
    data = []
    for yamlfile in filenames:
        yml = yaml.safe_load(open(yamlfile))
        n_atom = len(yml["structure"]["elements"])
        volume = float(yml["structure"]["volume"]) / n_atom
        for prop in yml["properties"]:
            temp = float(prop["temperature"])
            free_e = float(prop["free_energy"]) / n_atom
            entropy = float(prop["entropy"]) / n_atom
            # cv = float(prop["specific_heat"]) * EVtoJmol / n_atom
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="electron",
                free_energy=free_e,
                entropy=entropy,
                # heat_capacity=cv,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data


def _check_melting(log: np.ndarray):
    """Check whether MD simulation converges to a melting state."""
    if np.isclose(log[0, 2], 0.0):
        return False
    try:
        displacement_ratio = log[-1, 2] / log[0, 2]
        return displacement_ratio > 2.0
    except:
        return False


def load_ti_yamls(filenames: tuple[str], verbose: bool = False) -> list[GridPointData]:
    """Load polymlp_ti.yaml files."""
    data = []
    for yamlfile in filenames:
        res = load_thermodynamic_integration_yaml(yamlfile)
        temp, volume, free_e, ent, cv, eng, log = res
        if _check_melting(log):
            if verbose:
                message = yamlfile + " was eliminated (found to be in a melting state)."
                print(message, flush=True)
        else:
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="ti",
                free_energy=free_e,
                entropy=ent,
                # heat_capacity=cv,
                # energy=eng,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data


def compare_conditions(array1: np.ndarray, array2: np.ndarray):
    """Return indices with the same values in two arrays"""
    ids1, ids2 = [], []
    for i1, val in enumerate(array1):
        i2 = np.where(np.isclose(array2, val))[0]
        if len(i2) > 0:
            ids1.append(i1)
            ids2.append(i2[0])
    return np.array(ids1), np.array(ids2)


def get_common_grid(
    volumes1: np.ndarray,
    volumes2: np.ndarray,
    temperatures1: np.ndarray,
    temperatures2: np.ndarray,
):
    """Return common grid for two conditions."""
    ids1_v, ids2_v = compare_conditions(volumes1, volumes2)
    ids1_t, ids2_t = compare_conditions(temperatures1, temperatures2)
    return (ids1_v, ids1_t), (ids2_v, ids2_t)


def _calculate_harmonic_properties(
    res: Restart,
    path_fc2: str,
    temperatures: Optional[np.ndarray] = None,
    mesh: tuple = (10, 10, 10),
):
    """Calculate harmonic thermodynamic properties."""
    ph = Phonopy(structure_to_phonopy_cell(res.unitcell), res.supercell_matrix)
    ph.force_constants = read_fc2_from_hdf5(path_fc2)
    ph.run_mesh(mesh)
    ph.run_thermal_properties(temperatures=temperatures)
    tp_dict = ph.get_thermal_properties_dict()
    return tp_dict


def calculate_reference(grid_points: list[GridPointData], mesh: tuple = (10, 10, 10)):
    """Return reference properties.

    Harmonic phonon properties calculated with SSCHA FC2 and SSCHA frequencies
    at the lowest temperature are used as reference free energy, reference entropy,
    and reference heat capacity to fit properties with respect to temperature.
    """
    ref_id = 0
    for i, p in enumerate(grid_points):
        if p is not None:
            ref_id = i
            break

    if grid_points[ref_id].path_fc2 is None:
        raise RuntimeError("Reference state not found.")

    path_fc2 = grid_points[ref_id].path_fc2
    temperatures = np.array([p.temperature for p in grid_points])
    res = grid_points[ref_id].restart
    n_atom = len(res.unitcell.elements)

    tp_dict = _calculate_harmonic_properties(res, path_fc2, temperatures=temperatures)
    zip1 = zip(
        tp_dict["free_energy"],
        tp_dict["entropy"],
        tp_dict["heat_capacity"],
        grid_points,
    )
    for f, s, cv, point in zip1:
        if point is not None:
            point.reference_free_energy = f / EVtoKJmol / n_atom
            point.reference_entropy = s / EVtoJmol / n_atom
            point.reference_heat_capacity = cv / n_atom

    return grid_points


def calculate_harmonic_free_energies(
    grid_points: list[GridPointData],
    mesh: tuple = (10, 10, 10),
):
    """Return harmonic properties for grid points."""
    for point in grid_points:
        if point is not None:
            res = point.restart
            tp_dict = _calculate_harmonic_properties(res, point.path_fc2, mesh=mesh)
            n_atom = len(res.unitcell.elements)
            f = tp_dict["free_energy"][0]
            point.free_energy = f / EVtoKJmol / n_atom
            point.entropy = None
            point.heat_capacity = None

    return grid_points
