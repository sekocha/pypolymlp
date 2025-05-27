"""Utility functions for MD."""

from typing import Optional

import numpy as np
import yaml
from scipy.special.orthogonal import p_roots

from pypolymlp.calculator.md.ase_md import IntegratorASE


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
    delta_heat_capacity: Optional[float] = None,
    filename: str = "polymlp_ti.yaml",
):
    """Save results of thermodynamic integration."""
    np.set_printoptions(legacy="1.21")
    with open(filename, "w") as f:
        print("system:", integrator._atoms.symbols, file=f)
        print(file=f)

        print("units:", file=f)
        print("  volume:         angstrom3", file=f)
        print("  temperature:    K", file=f)
        print("  time_step:      fs", file=f)
        print("  energy:         eV/supercell", file=f)
        print("  heat_capacity:  J/K/mol (/Avogadro's number of atoms)", file=f)
        print(file=f)

        print("conditions:", file=f)
        print("  n_atom:     ", len(integrator._atoms.numbers), file=f)
        print("  volume:     ", integrator._atoms.get_volume(), file=f)
        print("  temperature:", integrator._temperature, file=f)
        print("  time_step:  ", integrator._time_step, file=f)
        print("  n_steps_eq: ", integrator._n_eq, file=f)
        print("  n_steps:    ", integrator._n_steps, file=f)
        print(file=f)

        print("properties:", file=f)
        print("  free_energy:         ", delta_free_energy, file=f)
        print("  average_energy:      ", integrator.average_energy, file=f)
        print("  delta_heat_capacity: ", delta_heat_capacity, file=f)
        print(file=f)

        print("  delta_energies:", file=f)
        for alpha, de, e, dis in log_ti:
            print("  - alpha:       ", alpha, file=f)
            print("    delta_e:     ", de, file=f)
            print("    energy:      ", e, file=f)
            print("    displacement:", np.round(dis, 5), file=f)
            print(file=f)


def load_thermodynamic_integration_yaml(filename: str = "polymlp_ti.yaml"):
    """Load results of thermodynamic integration.

    Return
    ------
    temperature: Temperature in K.
    volume: Volume in ang^3/atom.
    free_energy: Free energy difference in eV/atom.
    heat_capacity: Cv in J/K/mol (/Avogadro's number of atoms).
    log: Array of (alpha, delta_energy) in thermodynamic integration.
    """
    data = yaml.safe_load(open(filename))
    n_atom = int(data["conditions"]["n_atom"])
    temperature = float(data["conditions"]["temperature"])
    volume = float(data["conditions"]["volume"]) / n_atom
    prop = data["properties"]
    free_energy = float(prop["free_energy"]) / n_atom
    heat_capacity = float(prop["delta_heat_capacity"])
    log = [
        [float(d["alpha"]), float(d["delta_e"]), float(d["displacement"])]
        for d in prop["delta_energies"]
    ]
    return temperature, volume, free_energy, heat_capacity, np.array(log)
