"""Utility functions for initializing thermodynamic property calculation."""

from typing import Literal, Optional

import numpy as np
import yaml

from pypolymlp.calculator.compute_phonon import calculate_harmonic_properties_from_fc2
from pypolymlp.calculator.md.md_utils import load_thermodynamic_integration_yaml
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import GridPointData
from pypolymlp.core.units import EVtoJmol, EVtoKJmol


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
            grid.static_potential = res.static_potential
            grid.harmonic_free_energy = res.free_energy - res.anharmonic_energy
        else:
            grid.free_energy = None
            grid.entropy = None
        data.append(grid)

    return data


def load_electron_yamls(
    filenames: tuple[str],
    data_type: Literal["electron", "electron_ph"] = "electron",
) -> list[GridPointData]:
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
                data_type=data_type,
                free_energy=free_e,
                entropy=entropy,
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


def _is_success(eng: float, threshold: float = -100):
    """Check whether MD simulation is successfully finished."""
    if eng < threshold:
        return False
    return True


def load_ti_yamls(filenames: tuple[str], verbose: bool = False) -> list[GridPointData]:
    """Load polymlp_ti.yaml files."""
    data = []
    for yamlfile in filenames:
        res = load_thermodynamic_integration_yaml(yamlfile)
        temp, volume, free_e, ent, cv, eng, log = res
        if _is_success(eng):
            if _check_melting(log):
                if verbose:
                    message = " was eliminated (found to be in a melting state)."
                    print(yamlfile + message, flush=True)
            else:
                grid = GridPointData(
                    volume=volume,
                    temperature=temp,
                    data_type="ti",
                    free_energy=free_e,
                    entropy=ent,
                    # energy=eng,
                    path_yaml=yamlfile,
                )
                data.append(grid)
        else:
            if verbose:
                message = " was found to be a failed MD simulation."
                print(yamlfile + message, flush=True)
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


def calculate_reference(
    grid_points: list[GridPointData],
    temperatures: np.ndarray,
    mesh: tuple = (10, 10, 10),
):
    """Return reference properties.

    Harmonic phonon properties calculated with SSCHA FC2 and SSCHA frequencies
    at the lowest temperature are used as reference free energy, reference entropy,
    and reference heat capacity to fit properties with respect to temperature.
    """
    ref = None
    for point in grid_points:
        if point is not None and point.path_fc2 is not None:
            ref = point
            break

    if ref is None:
        raise RuntimeError("Reference state not found.")

    path_fc2, res = ref.path_fc2, ref.restart
    n_atom = len(res.unitcell.elements)

    tp_dict = _calculate_harmonic_properties(res, path_fc2, temperatures=temperatures)
    zip1 = zip(
        tp_dict["free_energy"],
        tp_dict["entropy"],
        tp_dict["heat_capacity"],
        grid_points,
    )
    # TODO: Reference entropy at the lowest temperature
    #       is not consistent with SSCHA entropy ?
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
