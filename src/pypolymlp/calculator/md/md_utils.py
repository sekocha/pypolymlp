"""Utility functions for MD."""

import numpy as np
from scipy.special.orthogonal import p_roots


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


def save_thermodynamic_integral(
    filename: str = "polymlp_ti.yaml",
    temperature: float = 300.0,
):
    """Save results of thermodynamic integration."""
