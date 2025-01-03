"""Utility functions for finding phase transition."""

import numpy as np
import scipy
import yaml


def parse_sscha_properties_yaml(yamlfile: str = "sscha_properties.yaml"):
    """Parse sscha_properties.yaml."""
    data = yaml.safe_load(open(yamlfile))
    helmholtz = [
        [d["temperature"], d["free_energy"]] for d in data["equilibrium_properties"]
    ]
    helmholtz = np.array(helmholtz)

    gibbs = dict()
    for d in data["gibbs_free_energies"]:
        temp = np.round(d["temperature"], 3)
        gibbs[temp] = np.array(d["values"])
    return helmholtz, gibbs


def _func(x, *args):
    return np.polyval(args[0], x)


def _fit_solve(f1: np.ndarray, f2: np.ndarray, f0: float = 0.0, order: int = 1):
    """Fit and solve delta f = 0."""
    z1 = np.polyfit(f1[:, 0], f1[:, 1], order)
    z2 = np.polyfit(f2[:, 0], f2[:, 1], order)
    coeffs = z1 - z2
    res = scipy.optimize.fsolve(_func, f0, args=coeffs)
    return res[0]


def find_transition(yaml1: str, yaml2: str):
    """Parse two sscha_properties.yaml files and find phase transition."""
    f1, _ = parse_sscha_properties_yaml(yaml1)
    f2, _ = parse_sscha_properties_yaml(yaml2)
    tc_linear_fit = _fit_solve(f1, f2, f0=0.0, order=1)
    tc_quartic_fit = _fit_solve(f1, f2, f0=tc_linear_fit, order=4)
    return tc_linear_fit, tc_quartic_fit


def compute_phase_boundary(yaml1: str, yaml2: str):
    """Parse two sscha_properties.yaml files and compute phase boundary."""
    _, g1 = parse_sscha_properties_yaml(yaml1)
    _, g2 = parse_sscha_properties_yaml(yaml2)

    for temp in g1.keys():
        try:
            g1_vals, g2_vals = g1[temp], g2[temp]
        except:
            continue

        # 1. p-G at T smoothing
        # 2. T-G at p smoothing
        # 3. tc at p (use find_transition)
        p_linear_fit = _fit_solve(g1_vals, g2_vals, f0=0.0, order=1)
        p_quartic_fit = _fit_solve(g1_vals, g2_vals, f0=p_linear_fit, order=4)
        print(temp, p_linear_fit, p_quartic_fit)
