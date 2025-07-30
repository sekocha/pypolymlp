"""Solvers for conjugate gradient."""

import copy
from typing import Optional

import numpy as np


def _eval_dot(
    vec: np.ndarray,
    xtx: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
):
    """Calculate (X.T @ X + alpha * I) @ vec."""
    if xtx is not None:
        return xtx @ vec + alpha * vec
    return x.T @ (x @ vec) + alpha * vec


def _eval_residual(
    coef: np.ndarray,
    xty: np.ndarray,
    xtx: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
):
    """Evaluate residual."""
    return xty - _eval_dot(coef, xtx=xtx, x=x, alpha=alpha)


def solver_cg(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    xtx: Optional[np.ndarray] = None,
    xty: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
    coef0: Optional[np.ndarray] = None,
    gtol: float = 1e-2,
    max_iter: int = 10000,
    verbose: bool = False,
):
    """Estimate MLP coefficients using conjugate gradient."""
    if x is None and xtx is None:
        raise RuntimeError("X or X.T @ X data not found.")
    if y is None and xty is None:
        raise RuntimeError("y or X.T @ y data not found.")

    if verbose:
        print("Use CG solver.", flush=True)
        print("conditions:", flush=True)
        print("- alpha:   ", alpha, flush=True)
        print("  gtol:    ", gtol, flush=True)
        print("  max_iter:", max_iter, flush=True)

    n_features = x.shape[1] if x is not None else xtx.shape[1]
    coef = np.zeros(n_features) if coef0 is None else copy.deepcopy(coef0)

    if xty is None:
        xty = x.T @ y

    residuals = _eval_residual(coef, xty, xtx=xtx, x=x, alpha=alpha)
    directions = copy.deepcopy(residuals)
    norm_residual = np.linalg.norm(residuals)

    norm_min, coef_min = 1e10, None
    for i in range(max_iter):
        norm_check = norm_residual / np.sqrt(n_features)
        if verbose:
            if i % 50 == 0:
                header = " CG iter. " + str(i) + ": residual ="
                print(header, np.round(norm_check, 5), flush=True)

        if i > 100 and norm_check < gtol:
            if verbose:
                print("CG successfully finished.", flush=True)
                header = " CG iter. " + str(i) + ": residual ="
                print(header, np.round(norm_check, 5), flush=True)
            break
        t = _eval_dot(directions, xtx=xtx, x=x, alpha=alpha)
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
