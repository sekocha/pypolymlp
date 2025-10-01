"""Utility functions for thermodynamic integral."""

from typing import Literal

import numpy as np
import scipy
import yaml

from pypolymlp.calculator.thermodynamics.fit_utils import Polyfit


def _is_success(eng: float, threshold: float = -100):
    """Check whether MD simulation is successfully finished."""
    if eng < threshold:
        return False
    return True


def _check_melting(disps: np.ndarray, threshold: float = 0.05):
    """Check whether MD simulation converges to a melting state."""
    disps = np.array(disps)
    if np.isclose(disps[0], 0.0):
        return np.zeros(len(disps), dtype=bool)

    ratio = disps / disps[0]
    bool1 = ratio > 2.0

    bool2 = [False]
    prev = disps[0]
    for d in disps[1:]:
        if abs(d - prev) > threshold:
            bool2.append(True)
        else:
            bool2.append(False)
            prev = d
    bool2 = np.array(bool2)

    return bool1 | bool2


def extrapolate_data(data: np.ndarray, max_order: int = 4):
    """Extrapolate TI data to alpha = 1.0 using polynomial fitting."""
    polyfit = Polyfit(data[:, 0], data[:, 1])
    polyfit.fit(max_order=max_order)
    return polyfit.eval(1.0)


def extrapolate_free_energy(
    data: np.ndarray,
    method: Literal["trapezoid", "simpson"] = "trapezoid",
    threshold: float = 0.7,
):
    """Extrapolate TI data to alpha = 1.0."""
    res = []
    for iend in range(data.shape[0]):
        alpha = data[iend, 0]
        if alpha > threshold:
            if method == "trapezoid":
                res1 = scipy.integrate.trapezoid(data[:iend, 1], data[:iend, 0])
            elif method == "simpson":
                res1 = scipy.integrate.simpson(data[:iend, 1], data[:iend, 0])
            res.append([alpha, res1])
    res = np.array(res)
    return extrapolate_data(res)


def load_thermodynamic_integration_yaml(
    filename: str = "polymlp_ti.yaml", verbose: bool = False
):
    """Load results of thermodynamic integration.

    Extrapolated properties to alpha = 1.0 are returned.

    Returns
    -------
    temperature: Temperature in K.
    volume: Volume in Angstroms^3/atom.
    free_energy: Free energy difference in eV/atom.
    entropy: Entropy difference in eV/K/atom.
    heat_capacity: Cv in J/K/mol (/Avogadro's number of atoms).
    energy: Energy difference in eV/atom.
    """
    data = yaml.safe_load(open(filename))
    n_atom = int(data["conditions"]["n_atom"])
    temperature = float(data["conditions"]["temperature"])
    volume = float(data["conditions"]["volume"]) / n_atom
    log = data["properties"]["delta_energies"]

    e_ref = float(log[0]["energy"])
    energy = (float(log[-1]["energy"]) - e_ref) / n_atom
    if not _is_success(energy):
        return None

    disps = [float(l["displacement"]) for l in log]
    is_melt = _check_melting(disps)
    if np.count_nonzero(is_melt) > len(is_melt) / 5:
        return None
    if is_melt[-1] and is_melt[-2]:
        return None

    log_active = [l1 for l1, bool1 in zip(log, is_melt) if not bool1]

    data_de = np.array([[float(l["alpha"]), float(l["delta_e"])] for l in log_active])
    free_energy = extrapolate_free_energy(data_de, method="trapezoid")
    free_energy /= n_atom

    data_e = np.array([[float(l["alpha"]), float(l["energy"])] for l in log_active])
    e_final = extrapolate_data(data_e)
    energy = (e_final - e_ref) / n_atom

    if np.isclose(temperature, 0.0):
        entropy = 0.0
    else:
        entropy = (energy - free_energy) / temperature

    if data["properties"]["delta_heat_capacity"] == "None":
        heat_capacity = None
    else:
        heat_capacity = float(data["properties"]["delta_heat_capacity"])

    if verbose:
        if np.any(is_melt):
            print("In", filename, flush=True)
        for l1, bool1 in zip(log, is_melt):
            if bool1:
                print(" alpha:", l1["alpha"], "was eliminated.", flush=True)

    return (temperature, volume, free_energy, entropy, heat_capacity, energy)
