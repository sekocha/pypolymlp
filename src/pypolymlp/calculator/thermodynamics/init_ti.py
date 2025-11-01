"""Functions for extracting properties from thermodynamic integration."""

from typing import Literal

import numpy as np
import scipy
import yaml

from pypolymlp.calculator.utils.fit_utils import Polyfit


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


def integrate(
    data: np.ndarray,
    method: Literal["trapezoid", "simpson", "romb"] = "trapezoid",
):
    """Integrate delta energy from TI."""
    if method == "trapezoid":
        free_energy = scipy.integrate.trapezoid(data[:, 1], data[:, 0])
    elif method == "simpson":
        free_energy = scipy.integrate.simpson(data[:, 1], x=data[:, 0])
    return free_energy


def _extrapolate_data(
    data: np.ndarray,
    max_order: int = 4,
    threshold: float = 0.7,
    verbose: bool = False,
):
    """Extrapolate TI data to alpha = 1.0 using polynomial fitting."""
    res = []
    for i in range(data.shape[0]):
        alpha = data[i, 0]
        if alpha > threshold:
            res.append([alpha, data[i, 1]])
    res = np.array(res)

    polyfit = Polyfit(res[:, 0], res[:, 1])
    polyfit.fit(max_order=max_order)
    if verbose:
        print("Extrapolate TI data.", flush=True)
        pred = polyfit.eval(res[:, 0])
        print("alpha, TI calculated, fitted", flush=True)
        for alpha, true_d, pred_d in zip(res[:, 0], res[:, 1], pred):
            print(alpha, true_d, pred_d, flush=True)
    return polyfit.eval(1.0)


def _get_free_energy(
    log_active: list,
    n_atom: int,
    extrapolation: bool = False,
    verbose: bool = False,
):
    """Calculate free energy from log."""
    data_de = np.array([[float(l["alpha"]), float(l["delta_e"])] for l in log_active])
    if extrapolation:
        de1 = _extrapolate_data(data_de, threshold=0.9, max_order=2, verbose=verbose)
        data_de = np.vstack((data_de, [1.0, de1]))

    # free_energy = integrate(data_de, method="trapezoid")
    free_energy = integrate(data_de, method="simpson")
    free_energy /= n_atom
    return free_energy


def _get_energy(
    log_active: list,
    n_atom: int,
    extrapolation: bool = False,
    verbose: bool = False,
):
    """Calculate potential energy from log."""
    data_e = np.array([[float(l["alpha"]), float(l["energy"])] for l in log_active])
    energy0 = data_e[1, 0]
    if extrapolation:
        energy = _extrapolate_data(
            data_e,
            max_order=2,
            threshold=0.9,
            verbose=verbose,
        )
    else:
        energy = data_e[1, -1]
    energy = (energy - energy0) / n_atom
    return energy


def load_ti_yaml(
    filename: str = "polymlp_ti.yaml",
    extrapolation: bool = False,
    verbose: bool = False,
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
    if is_melt[-1] and is_melt[-2] and is_melt[-3]:
        return None

    log_active = [l1 for l1, bool1 in zip(log, is_melt) if not bool1]
    # free_energy = float(data["properties"]["free_energy"])
    free_energy = _get_free_energy(
        log_active,
        n_atom,
        extrapolation=extrapolation,
        verbose=verbose,
    )
    energy = _get_energy(
        log_active,
        n_atom,
        extrapolation=extrapolation,
        verbose=verbose,
    )

    # data_de = np.array([[float(l["alpha"]), float(l["delta_e"])] for l in log_active])
    # if extrapolation:
    #     de1 = _extrapolate_data(data_de, threshold=0.9, max_order=2, verbose=verbose)
    #     data_de = np.vstack((data_de, [1.0, de1]))

    # # free_energy = integrate(data_de, method="trapezoid")
    # free_energy = integrate(data_de, method="simpson")
    # free_energy /= n_atom

    # data_e = np.array([[float(l["alpha"]), float(l["energy"])] for l in log_active])
    # energy0 = log_active[0]
    # if extrapolation:
    #     energy1 = _extrapolate_data(
    #         data_e,
    #         max_order=2,
    #         threshold=0.9,
    #         verbose=verbose,
    #     )
    # else:
    #     energy = data_e[-1]
    # energy = (energy - energy0) / n_atom

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

    return (temperature, volume, free_energy, energy, entropy, heat_capacity)


# def load_ti_yamls_fit(
#    filenames: tuple[str],
#    extrapolation: bool = False,
#    verbose: bool = False,
# ) -> list[GridPointData]:
#    """Load polymlp_ti.yaml files."""
#    data = []
#    for yamlfile in filenames:
#        res = load_thermodynamic_integration_yaml(
#            yamlfile,
#            extrapolation=extrapolation,
#            verbose=verbose,
#        )
#        if res is not None:
#            temp, volume, free_e, energy, entropy, cv = res
#            grid = GridPointData(
#                volume=volume,
#                temperature=temp,
#                data_type="ti",
#                free_energy=free_e,
#                entropy=entropy,
#                energy=energy,
#                heat_capacity=cv,
#                path_yaml=yamlfile,
#            )
#            data.append(grid)
#        else:
#            if verbose:
#                message = " was eliminated (failed or in a melting state)."
#                print(yamlfile + message, flush=True)
#    return data
