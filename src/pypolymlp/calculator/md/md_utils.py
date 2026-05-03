"""Utility functions for MD."""

import glob

import numpy as np
import yaml
from scipy.special import p_roots

# import os
# from typing import Optional


# from pypolymlp.calculator.md.ase_md import IntegratorASE
# from pypolymlp.calculator.utils.io_utils import print_pot


def find_reference(path_fc2: str, target_temperature: float):
    """Find reference FC2 automatically.

    The FC2 state at the lowest temperature is regareded as
    the reference state used as reference state for free energy calculations.
    """
    reference = None
    temp_min = 1e10
    for fc2hdf5 in sorted(glob.glob(path_fc2 + "/*/fc2.hdf5")):
        path = "/".join(fc2hdf5.split("/")[:-1])
        yamlname = path + "/sscha_results.yaml"
        data = yaml.safe_load(open(yamlname))
        converge = data["status"]["converge"]
        if not converge:
            continue
        imaginary = data["status"]["imaginary"]
        if imaginary:
            continue

        temp = float(data["parameters"]["temperature"])
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
