"""Solvers for stochastic gradient descent."""

from typing import Optional

import numpy as np


def solver_sgd(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    xtx: Optional[np.ndarray] = None,
    xty: Optional[np.ndarray] = None,
    alphas: tuple = (1e-3, 1e-2, 1e-1),
    verbose: bool = False,
):
    """Estimate MLP coefficients using stochastic gradient descent."""
    pass
