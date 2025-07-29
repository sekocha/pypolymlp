"""Solvers for conjugate gradient."""

import copy
from typing import Optional

import numpy as np


def _eval_dot(x: np.ndarray, alpha: float, vec: np.ndarray):
    """Calculate (X.T @ X + alpha * I) @ vec."""
    return x.T @ (x @ vec) + alpha * vec


def _eval_residual(coef: np.ndarray, x: np.ndarray, xty: np.ndarray, alpha: float):
    """Evaluate residual."""
    return xty - _eval_dot(x, alpha, coef)


def solver_cg(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    coef0: Optional[np.ndarray] = None,
    gtol: float = 1e-2,
    max_iter: int = 10000,
    verbose: bool = False,
):
    """Estimate MLP coefficients using conjugate gradient."""

    if verbose:
        print("Use CG solver.", flush=True)
        print("conditions:", flush=True)
        print("- alpha:   ", alpha, flush=True)
        print("  gtol:    ", gtol, flush=True)
        print("  max_iter:", max_iter, flush=True)

    coef = np.zeros(x.shape[1]) if coef0 is None else copy.deepcopy(coef0)
    xty = x.T @ y
    residuals = _eval_residual(coef, x, xty, alpha)
    directions = copy.deepcopy(residuals)
    norm_residual = np.linalg.norm(residuals)

    norm_min, coef_min = 1e10, None
    for i in range(max_iter):
        if verbose:
            if i % 50 == 0:
                header = " CG iter. " + str(i) + ": residual ="
                print(header, np.round(norm_residual / x.shape[1], 5), flush=True)

        if i > 0 and norm_residual / x.shape[1] < gtol:
            if verbose:
                print("CG successfully finished.", flush=True)
                header = " CG iter. " + str(i) + ": residual ="
                print(header, np.round(norm_residual / x.shape[1], 5), flush=True)
            break
        t = _eval_dot(x, alpha, directions)
        learning_rate = (norm_residual**2) / (t @ directions)
        coef += learning_rate * directions
        residuals -= learning_rate * t
        norm_residual_new = np.linalg.norm(residuals)
        beta = (norm_residual_new**2) / (norm_residual**2)
        directions = residuals + beta * directions
        norm_residual = norm_residual_new

        if norm_min > norm_residual:
            norm_min = norm_residual
            coef_min = coef

    return coef_min
