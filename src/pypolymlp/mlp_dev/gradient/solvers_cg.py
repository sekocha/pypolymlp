"""Solvers for stochastic gradient descent."""

import copy
from typing import Optional

import numpy as np


def _func(x, xty, coef, grad, alpha, beta, learning_rate):
    grad_add = x.T @ (x @ coef) + alpha * coef - xty
    grad_new = beta * grad + (1 - beta) * (-grad_add)
    coef += learning_rate * grad_new
    grad = grad_new
    return coef, grad


def _eval_residual(coef: np.ndarray, x: np.ndarray, xty: np.ndarray, alpha: float):
    """Evaluate residual."""
    return xty - alpha * coef - x.T @ (x @ coef)


def solver_cg(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    coef0: Optional[np.ndarray] = None,
    gtol: float = 1e-1,
    max_iter: int = 100000,
    verbose: bool = False,
):
    """Estimate MLP coefficients using conjugate gradient."""

    if verbose:
        print("Use CG solver.", flush=True)
        print("- alpha:", alpha, flush=True)

    if coef0 is None:
        coef = np.zeros(x.shape[1])
    else:
        coef = coef0

    xty = x.T @ y

    residuals = _eval_residual(coef, x, xty, alpha)
    directions = copy.deepcopy(residuals)
    norm_residual = np.linalg.norm(residuals)

    for i in range(max_iter):
        if norm_residual < gtol:
            break
        t = x.T @ (x @ directions)
        learning_rate = (norm_residual**2) / (t @ directions)
        coef += learning_rate * directions
        print(coef)
        residuals -= learning_rate * t
        norm_residual_new = np.linalg.norm(residuals)
        beta = (norm_residual_new**2) / (norm_residual**2)

        directions = residuals + beta * directions
        norm = norm_residual
        print(norm, residuals)

    return coef
