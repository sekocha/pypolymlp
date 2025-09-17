"""Utility functions for MD."""

import glob
import os
from typing import Optional

import numpy as np
import yaml
from scipy.special.orthogonal import p_roots

from pypolymlp.calculator.compute_phonon import calculate_harmonic_properties_from_fc2
from pypolymlp.calculator.md.ase_md import IntegratorASE
from pypolymlp.core.units import EVtoKJmol


def get_p_roots(n: int = 10, a: float = -1.0, b: float = 1.0):
    """Compute sample points and weights for Gauss-Legendre quadrature."""
    x, w = p_roots(n)
    x_rev = (0.5 * (b - a)) * x + (0.5 * (a + b))
    return x_rev, w


def calc_integral(
    w: np.ndarray,
    f: np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
):
    """Compute integral from sample points using Gauss-Legendre quadrature."""
    return (0.5 * (b - a)) * w @ np.array(f)


def save_thermodynamic_integration_yaml(
    integrator: IntegratorASE,
    delta_free_energy: float,
    log_ti: np.array,
    reference: dict,
    delta_heat_capacity: Optional[float] = None,
    filename: str = "polymlp_ti.yaml",
):
    """Save results of thermodynamic integration."""
    np.set_printoptions(legacy="1.21")

    tp_dict = calculate_harmonic_properties_from_fc2(
        unitcell=reference["unitcell"],
        supercell_matrix=reference["supercell_matrix"],
        path_fc2=reference["fc2_file"],
        mesh=(10, 10, 10),
        temperatures=[integrator._temperature],
    )
    n_unitcell = np.linalg.det(reference["supercell_matrix"])
    ref_free_energy = tp_dict["free_energy"][0] * n_unitcell / EVtoKJmol
    total_free_energy = integrator.static_energy + ref_free_energy + delta_free_energy

    with open(filename, "w") as f:
        print("system:", integrator._atoms.symbols, file=f)
        print(file=f)

        print("units:", file=f)
        print("  volume:         angstrom3", file=f)
        print("  temperature:    K", file=f)
        print("  time_step:      fs", file=f)
        print("  energy:         eV/supercell", file=f)
        print("  entropy:        eV/K/supercell", file=f)
        print("  heat_capacity:  J/K/mol (/Avogadro's number of atoms)", file=f)
        print(file=f)

        print("conditions:", file=f)
        print("  n_atom:     ", len(integrator._atoms.numbers), file=f)
        print("  volume:     ", integrator._atoms.get_volume(), file=f)
        print("  temperature:", integrator._temperature, file=f)
        print("  time_step:  ", integrator._time_step, file=f)
        print("  n_steps_eq: ", integrator._n_eq, file=f)
        print("  n_steps:    ", integrator._n_steps, file=f)
        print("  references: ", os.path.abspath(reference["fc2_file"]), file=f)
        print(file=f)

        delta_entropy = 0.0
        if not np.isclose(integrator._temperature, 0.0):
            e_final = float(log_ti[-1][2])
            e_ref = float(log_ti[0][2])
            de = e_final - e_ref
            delta_entropy = (de - delta_free_energy) / integrator._temperature

        print("properties:", file=f)
        print("  free_energy:             ", delta_free_energy, file=f)
        print("  entropy:                 ", delta_entropy, file=f)
        print("  average_potential_energy:", integrator.average_energy, file=f)
        print("  average_total_energy:    ", integrator.average_total_energy, file=f)
        print("  delta_heat_capacity:     ", delta_heat_capacity, file=f)

        print(file=f)
        print("  static_potential_energy: ", integrator.static_energy, file=f)
        print("  reference_free_energy:   ", ref_free_energy, file=f)
        print("  total_free_energy:       ", total_free_energy, file=f)
        print(file=f)

        print("  delta_energies:", file=f)
        for alpha, de, e, total_e, dis in log_ti:
            print("  - alpha:       ", alpha, file=f)
            print("    delta_e:     ", de, file=f)
            print("    energy:      ", e, file=f)
            print("    total_energy:", total_e, file=f)
            print("    displacement:", np.round(dis, 5), file=f)
            print(file=f)


def load_thermodynamic_integration_yaml(filename: str = "polymlp_ti.yaml"):
    """Load results of thermodynamic integration.

    Return
    ------
    temperature: Temperature in K.
    volume: Volume in ang^3/atom.
    free_energy: Free energy difference in eV/atom.
    entropy: Entropy difference in eV/K/atom.
    heat_capacity: Cv in J/K/mol (/Avogadro's number of atoms).
    log: Array of (alpha, delta_energy) in thermodynamic integration.
    """
    data = yaml.safe_load(open(filename))
    n_atom = int(data["conditions"]["n_atom"])
    temperature = float(data["conditions"]["temperature"])
    volume = float(data["conditions"]["volume"]) / n_atom
    prop = data["properties"]
    free_energy = float(prop["free_energy"]) / n_atom

    e_final = float(prop["delta_energies"][-1]["energy"])
    e_ref = float(prop["delta_energies"][0]["energy"])
    energy = (e_final - e_ref) / n_atom

    if "entropy" in prop:
        entropy = float(prop["entropy"]) / n_atom
    else:
        if np.isclose(temperature, 0.0):
            entropy = 0.0
        else:
            entropy = (energy - free_energy) / temperature

    if prop["delta_heat_capacity"] == "None":
        heat_capacity = None
    else:
        heat_capacity = float(prop["delta_heat_capacity"])
    log = [
        [float(d["alpha"]), float(d["delta_e"]), float(d["displacement"])]
        for d in prop["delta_energies"]
    ]
    return (
        temperature,
        volume,
        free_energy,
        entropy,
        heat_capacity,
        energy,
        np.array(log),
    )


def find_reference(path_fc2: str, target_temperature: float):
    """Find reference FC2 automatically."""
    reference = None
    temp_min = 1e10
    for fc2hdf5 in sorted(glob.glob(path_fc2 + "/*/fc2.hdf5")):
        path = "/".join(fc2hdf5.split("/")[:-1])
        yamlname = path + "/sscha_results.yaml"
        data = yaml.safe_load(open(yamlname))
        temp = float(data["parameters"]["temperature"])
        converge = data["status"]["converge"]
        imaginary = data["status"]["imaginary"]
        success = True if converge and not imaginary else False
        if success:
            if np.isclose(temp, 0.0):
                temp_min = 0.0
                reference = fc2hdf5
                break
            else:
                if temp < temp_min:
                    temp_min = temp
                    reference = fc2hdf5

    if reference is None:
        raise RuntimeError("No reference state found.")
    if target_temperature + 1e-8 < temp_min:
        raise RuntimeError("Target temperature is lower than reference temperature.")
    return reference
