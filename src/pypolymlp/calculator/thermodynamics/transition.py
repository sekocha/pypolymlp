"""Utility functions for finding phase transition."""

import numpy as np

from pypolymlp.calculator.thermodynamics.fit_utils import Polyfit, fit_solve_poly


def find_transition(f1: np.ndarray, f2: np.ndarray):
    """Find phase transition from F(T) data in two thermodynamics.yaml files.

    Parameters
    ----------
    f1: First data. (temperature, Helmholtz free energy)
    f2: Second data. (temperature, Helmholtz free energy)
    """
    tc_linear_fit = fit_solve_poly(f1, f2, f0=0.0, order=1)
    tc_polyfit = fit_solve_poly(f1, f2, f0=tc_linear_fit)
    return tc_polyfit


def _fit_pressure_gibbs(
    gibbs: np.ndarray, temperatures: np.ndarray, pressures_eval: np.ndarray
):
    """Fit p-G data at T and set T-G data at p."""
    g_fit_evals = [[] for p in pressures_eval]
    for temp, g in zip(temperatures, gibbs):
        p = Polyfit(g[:, 0], g[:, 1]).fit()
        gvals = p.eval(pressures_eval)
        for i, v in enumerate(gvals):
            g_fit_evals[i].append([temp, v])
    return g_fit_evals


def compute_phase_boundary(
    gibbs1: np.ndarray,
    temperatures1: np.ndarray,
    gibbs2: np.ndarray,
    temperatures2: np.ndarray,
):
    """Compute phase boundary from G(p, T) data in two thermodynamics.yaml files."""

    p1_max = max([np.max(g[:, 0]) for g in gibbs1]) * 0.8
    p2_max = max([np.max(g[:, 0]) for g in gibbs2]) * 0.8
    p_max = min(p1_max, p2_max)
    pressures = np.arange(-3.0, p_max, 0.25)

    g1_fit_evals = _fit_pressure_gibbs(gibbs1, temperatures1, pressures)
    g2_fit_evals = _fit_pressure_gibbs(gibbs2, temperatures2, pressures)

    # fit T-G at p and estimate Tc
    boundary = []
    for tg1, tg2, press in zip(g1_fit_evals, g2_fit_evals, pressures):
        if press > -1e-5:
            tg1, tg2 = np.array(tg1), np.array(tg2)
            tc_linear_fit = fit_solve_poly(tg1, tg2, f0=0.0, order=1)
            tc_polyfit = fit_solve_poly(tg1, tg2, f0=tc_linear_fit, max_order=3)
            if tc_polyfit > 0:
                boundary.append([press, tc_polyfit])

    return np.array(boundary)
