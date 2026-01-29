"""Utility functions for finding phase transition."""

import numpy as np

from pypolymlp.calculator.utils.fit_utils import Polyfit, fit_solve, fit_solve_poly


def find_transition(f1: np.ndarray, f2: np.ndarray):
    """Find phase transition from F(T) data in two thermodynamics.yaml files.

    Parameters
    ----------
    f1: First data. (temperature, Helmholtz free energy)
    f2: Second data. (temperature, Helmholtz free energy)
    """
    tc0 = fit_solve_poly(f1, f2, f0=0.0, order=1)
    tc_fit = fit_solve(f1, f2, f0=tc0, k=1)
    return tc_fit


def _fit_pressure_gibbs(
    gibbs: np.ndarray,
    temperatures: np.ndarray,
    pressures_eval: np.ndarray,
    max_order: int = 4,
):
    """Fit p-G data at T and set T-G data at p."""
    g_fit_evals = [[] for p in pressures_eval]
    for temp, g in zip(temperatures, gibbs):
        p = Polyfit(g[:, 0], g[:, 1]).fit(max_order=max_order)
        gvals = p.eval(pressures_eval)
        for i, v in enumerate(gvals):
            g_fit_evals[i].append([temp, v])
    return g_fit_evals


def compute_phase_boundary(
    gibbs1: np.ndarray,
    temperatures1: np.ndarray,
    gibbs2: np.ndarray,
    temperatures2: np.ndarray,
    pressure_interval: float = 0.25,
    fit_gibbs_max_order: int = 4,
):
    """Compute phase boundary from G(p, T) data in two thermodynamics.yaml files."""

    p1_max = max([np.max(g[:, 0]) for g in gibbs1]) * 0.8
    p2_max = max([np.max(g[:, 0]) for g in gibbs2]) * 0.8
    p_max = min(p1_max, p2_max)
    pressures = np.arange(-3.0, p_max, pressure_interval)

    # fit p-G at T
    g1_fit_evals = _fit_pressure_gibbs(
        gibbs1,
        temperatures1,
        pressures,
        max_order=fit_gibbs_max_order,
    )
    g2_fit_evals = _fit_pressure_gibbs(
        gibbs2,
        temperatures2,
        pressures,
        max_order=fit_gibbs_max_order,
    )

    # fit T-G at p and estimate Tc
    boundary = []
    for tg1, tg2, press in zip(g1_fit_evals, g2_fit_evals, pressures):
        if press > -1e-5:
            tg1, tg2 = np.array(tg1), np.array(tg2)
            tc_fit = find_transition(tg1, tg2)
            if tc_fit > 0:
                boundary.append([press, tc_fit])

    return np.array(boundary)
