"""Utility functions for initializing thermodynamic property calculation."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.compute_phonon import calculate_harmonic_properties_from_fc2
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import GridPointData
from pypolymlp.core.units import EVtoJmol, EVtoKJmol


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
