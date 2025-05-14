"""Utility functions for MD."""

import numpy as np
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
    delta_energies: np.array,
    filename: str = "polymlp_ti.yaml",
):
    """Save results of thermodynamic integration."""
    with open(filename, "w") as f:
        print("system:", integrator._atoms.symbols, file=f)
        print(file=f)

        print("units:", file=f)
        print("  volume:         angstrom3", file=f)
        print("  temperature:    K", file=f)
        print("  time_step:      fs", file=f)
        print("  energy:         eV/supercell", file=f)
        print("  heat_capacity:  eV/K", file=f)
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
        print("  free_energy:", delta_free_energy, file=f)
        print("  delta_energies:", file=f)
        for alpha, de in delta_energies:
            print("  - alpha:  ", alpha, file=f)
            print("    delta_e:", de, file=f)


#        print("  heat_capacity: ", self._heat_capacity, file=f)
