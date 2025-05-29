"""Utility functions for finding phase transition."""

from collections import defaultdict
from typing import Optional

import numpy as np
import scipy
import yaml

from pypolymlp.calculator.sscha.utils.lsq import polyfit


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


def _func_poly(x, *args):
    return np.polyval(args[0], x)


def _fit_poly(
    f1: np.ndarray,
    f2: np.ndarray,
    order: Optional[int] = None,
    max_order: int = 6,
):
    """Fit data using a polynomial."""
    if order is not None:
        z1 = np.polyfit(f1[:, 0], f1[:, 1], order)
        z2 = np.polyfit(f2[:, 0], f2[:, 1], order)
    else:
        z1 = polyfit(f1[:, 0], f1[:, 1], max_order=max_order, verbose=True)
        z2 = polyfit(f2[:, 0], f2[:, 1], max_order=max_order, verbose=True)
        len_diff = len(z1) - len(z2)
        if len_diff > 0:
            z2 = np.hstack([np.zeros(len_diff), z2])
        elif len_diff < 0:
            z1 = np.hstack([np.zeros(-len_diff), z1])
    return z1, z2


def _fit_solve_poly(
    f1: np.ndarray,
    f2: np.ndarray,
    f0: float = 0.0,
    order: Optional[int] = None,
    max_order: int = 6,
):
    """Fit and solve delta f = 0."""
    z1, z2 = _fit_poly(f1, f2, order=order, max_order=max_order)
    coeffs = z1 - z2
    res = scipy.optimize.fsolve(_func_poly, f0, args=coeffs)
    return res[0]


def _func_spline(x, *args):
    sp1, sp2 = args
    return sp1(x) - sp2(x)


def _fit_solve_spline(f1: np.ndarray, f2: np.ndarray, f0: float = 0.0, k: int = 3):
    """Fit and solve delta f = 0."""
    sp1 = scipy.interpolate.make_interp_spline(f1[:, 0], f1[:, 1], k=k)
    sp2 = scipy.interpolate.make_interp_spline(f2[:, 0], f2[:, 1], k=k)
    args = sp1, sp2
    res = scipy.optimize.fsolve(_func_spline, f0, args=args)
    return res[0]


def find_transition(yaml1: str, yaml2: str):
    """Parse two sscha_properties.yaml files and find phase transition."""
    f1, _ = parse_sscha_properties_yaml(yaml1)
    f2, _ = parse_sscha_properties_yaml(yaml2)
    tc_linear_fit = _fit_solve_poly(f1, f2, f0=0.0, order=1)
    tc_polyfit = _fit_solve_poly(f1, f2, f0=tc_linear_fit)
    return tc_linear_fit, tc_polyfit


def compute_phase_boundary(yaml1: str, yaml2: str):
    """Parse two sscha_properties.yaml files and compute phase boundary."""
    _, g1 = parse_sscha_properties_yaml(yaml1)
    _, g2 = parse_sscha_properties_yaml(yaml2)

    p1_max = max([np.max(g[:, 0]) for g in g1.values()]) * 0.5
    p2_max = max([np.max(g[:, 0]) for g in g2.values()]) * 0.5
    p_max = min(p1_max, p2_max)

    pressures = np.arange(0, p_max, 0.5)
    g1_fit, g2_fit = defaultdict(list), defaultdict(list)
    for temp in g1.keys():
        try:
            g1_vals, g2_vals = g1[temp], g2[temp]
        except:
            continue

        # 1. fit p-G at T
        z1, z2 = _fit_poly(g1_vals, g2_vals)
        vals1 = np.polyval(z1, pressures)
        vals2 = np.polyval(z2, pressures)

        # 2. T-G data at p
        for p, v in zip(pressures, vals1):
            g1_fit[np.round(p, 5)].append([temp, v])
        for p, v in zip(pressures, vals2):
            g2_fit[np.round(p, 5)].append([temp, v])

    # 3. fit T-G at p and estimate Tc
    boundary = []
    for press in pressures:
        tg1 = np.array(g1_fit[np.round(press, 5)])
        tg2 = np.array(g2_fit[np.round(press, 5)])
        tc_linear_fit = _fit_solve_poly(tg1, tg2, f0=0.0, order=1)
        tc_polyfit = _fit_solve_poly(tg1, tg2, f0=tc_linear_fit, max_order=3)
        boundary.append([press, tc_polyfit])
    return boundary
