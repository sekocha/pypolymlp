"""Utility functions for fitting thermodynamic properties."""

import itertools
from typing import Optional

import numpy as np

from pypolymlp.core.utils import rmse


def loocv(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate leave-one-out cross validation score."""
    residuals = y_pred - y_true
    h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    squared_errors = (residuals / (1 - h)) ** 2
    loocv = np.sqrt(np.mean(squared_errors))
    return loocv


class Polyfit:
    """Class for fitting properties to a polynomial function."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Init method."""
        self._x = x
        self._y = y

    def eval(self, x):
        pass

    def eval_derivative(self, x):
        pass

    def fit(
        self,
        order: Optional[int] = None,
        max_order: int = 4,
        intercept: bool = True,
        add_sqrt: bool = False,
    ):
        """Fit data to polynomial functions.

        If order is None, the optimal value of order will be automatically
        determined by minimizing the leave-one-out cross validation score.
        """
        x = self._x
        y = self._y
        orders = list(range(2, max_order + 1)) if order is None else [order]
        sqrts = [True, False] if add_sqrt is None else [add_sqrt]
        if len(orders) == 1 and len(sqrts) == 1:
            best_order = orders[0]
            best_add_sqrt = sqrts[0]
        else:
            min_loocv = 1e10
            best_order, best_add_sqrt = None, None
            params = list(itertools.product(sqrts, orders))
            for add_sqrt, order in params:
                (poly_coeffs, y_pred, y_rmse), X = self._polyfit_single(
                    x,
                    y,
                    order,
                    intercept=intercept,
                    add_sqrt=add_sqrt,
                )
                cv = loocv(X, y, y_pred)
                if min_loocv > cv:
                    min_loocv = cv
                    best_order = order
                    best_add_sqrt = add_sqrt

        (poly_coeffs, y_pred, y_rmse), _ = self._polyfit_single(
            x,
            y,
            best_order,
            intercept=intercept,
            add_sqrt=best_add_sqrt,
        )
        if not intercept:
            poly_coeffs = list(poly_coeffs)
            poly_coeffs.append(0.0)
            poly_coeffs = np.array(poly_coeffs)

        return (poly_coeffs, y_pred, y_rmse), (best_order, best_add_sqrt)

    def _fit_single(self, order: int, intercept: bool = True, add_sqrt: bool = False):
        """Fit data to a single polynomial with a given order."""
        X = []
        if add_sqrt:
            X.append(np.sqrt(self._x))
        for power in np.arange(order, 0, -1, dtype=int):
            X.append(self._x**power)
        if intercept:
            X.append(np.ones(self._x.shape))
        X = np.array(X).T

        poly_coeffs = np.linalg.solve(X.T @ X, X.T @ self._y)
        y_pred = X @ poly_coeffs
        y_rmse = rmse(self._y, y_pred)
        return (poly_coeffs, y_pred, y_rmse), X
