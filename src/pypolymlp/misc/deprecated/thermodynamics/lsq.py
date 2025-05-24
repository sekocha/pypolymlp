"""Utility functions for fitting data."""

from typing import Optional

import numpy as np


def loocv(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate leave-one-out cross validation score."""
    residuals = y_pred - y_true
    h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    squared_errors = (residuals / (1 - h)) ** 2
    loocv = np.sqrt(np.mean(squared_errors))
    return loocv


def polyfit(
    x: np.ndarray,
    y: np.ndarray,
    order: Optional[int] = None,
    max_order: int = 4,
    verbose: bool = False,
):
    """Fit data to a polynomial.

    If order is None, the optimal value of order will be automatically
    determined by minimizing the leave-one-out cross validation score.
    """
    orders = list(range(2, max_order + 1)) if order is None else [order]
    if len(orders) == 1:
        best_order = orders[0]
    else:
        min_loocv = 1e10
        best_order = None
        if verbose:
            print("Find optimal polynomial.", flush=True)
        for order in orders:
            poly_coeffs = np.polyfit(x, y, order)
            y_pred = np.polyval(poly_coeffs, x)
            X = [x**power for power in np.arange(order, -1, -1, dtype=int)]
            X = np.array(X).T
            cv = loocv(X, y, y_pred)
            if min_loocv > cv:
                min_loocv = cv
                best_order = order
            if verbose:
                print("order:", order, "loocv:", cv, flush=True)

    poly_coeffs = np.polyfit(x, y, best_order)
    return poly_coeffs
