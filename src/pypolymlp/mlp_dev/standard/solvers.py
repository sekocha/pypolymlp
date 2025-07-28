"""Solvers."""

from typing import Optional

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs


def solve_linear_equation(A: np.ndarray, b: np.ndarray, verbose: bool = False):
    """Solve Ax = b.

    Alternative implementations using numpy and scipy are as follows.
    (Alternative 1)
    x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
    (Alternative 2)
    x = np.linalg.solve(A, b)
    (Alternative 3)
    L = np.linalg.cholesky(A)
    x = np.linalg.solve(L.T, np.linalg.solve(L, b))
    (Alternative 4)
    x, exit_code = scipy.sparse.linalg.cg(A, b, rtol=1e-8)
    """
    (posv,) = get_lapack_funcs(("posv",), (A, b))
    if A.flags["C_CONTIGUOUS"]:
        _, x, info = posv(
            A.T,
            b,
            lower=False,
            overwrite_a=False,
            overwrite_b=False,
        )
    else:
        _, x, info = posv(
            A,
            b,
            lower=False,
            overwrite_a=False,
            overwrite_b=False,
        )
    if not info == 0:
        if verbose:
            comment = "Error: The factorization could not be completed."
            print(" ", comment, flush=True)
        x = np.ones(x.shape) * 1e30
    return x


def solver_ridge(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    xtx: Optional[np.ndarray] = None,
    xty: Optional[np.ndarray] = None,
    alphas: tuple = (1e-3, 1e-2, 1e-1),
    verbose: bool = False,
):
    """Estimate MLP coefficients using ridge regression."""
    if verbose:
        print("Regression: Use standard ridge solver.", flush=True)

    if xtx is None or xty is None:
        if verbose:
            print("Regression: compute X.T @ X and X.T @ y.", flush=True)
        xtx = x.T @ x
        xty = x.T @ y

    if verbose:
        print("Regression: cholesky decomposition.", flush=True)

    n_features = xtx.shape[0]
    coefs_array = np.zeros((n_features, len(alphas)))
    alpha_prev = 0.0
    for i, alpha in enumerate(alphas):
        if verbose:
            print("- alpha:", alpha, flush=True)
        add = alpha - alpha_prev
        if verbose:
            print("  Compute X.T @ X + alpha @ I", flush=True)
        xtx.flat[:: n_features + 1] += add
        if verbose:
            print("  Solve linear equation", flush=True)
        coefs_array[:, i] = solve_linear_equation(xtx, xty)
        alpha_prev = alpha
    xtx.flat[:: n_features + 1] -= alpha
    return coefs_array
