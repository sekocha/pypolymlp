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
        self._coeffs = None
        self._error = None
        self._order = None
        self._add_sqrt = None

    def eval(self, x: float):
        """Evaluate value of fitted polynomial at given x."""
        return np.polyval(self._coeffs, x)

    def eval_derivative(self, x: float):
        """Evaluate derivative of fitted polynomial at given x."""
        deriv = self._get_poly_deriv()
        val = np.polyval(deriv, x)
        if self._add_sqrt:
            val += 0.5 * self._coeffs[0] * np.power(x, -0.5)
        return val

    def _get_poly_deriv(self):
        """Return derivatives of coefficients from polynomial fits."""
        deriv = self._coeffs * np.arange(len(self._coeffs) - 1, -1, -1, dtype=int)
        return deriv[:-1]

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
        orders = list(range(2, max_order + 1)) if order is None else [order]
        sqrts = [True, False] if add_sqrt is None else [add_sqrt]
        if len(orders) == 1 and len(sqrts) == 1:
            best_order, best_add_sqrt = orders[0], sqrts[0]
        else:
            min_loocv, best_order, best_add_sqrt = 1e10, None, None
            params = list(itertools.product(sqrts, orders))
            for add_sqrt, order in params:
                res = self._fit_single(order, intercept=intercept, add_sqrt=add_sqrt)
                (_, y_pred, _), X = res
                cv = loocv(X, self._y, y_pred)
                if min_loocv > cv:
                    min_loocv = cv
                    best_order, best_add_sqrt = order, add_sqrt

        (coeffs, y_pred, y_rmse), _ = self._fit_single(
            best_order,
            intercept=intercept,
            add_sqrt=best_add_sqrt,
        )
        if not intercept:
            coeffs = list(coeffs)
            coeffs.append(0.0)
            coeffs = np.array(coeffs)

        self._coeffs = coeffs
        self._error = y_rmse
        self._order = best_order
        self._add_sqrt = best_add_sqrt
        return self

    def _fit_single(self, order: int, intercept: bool = True, add_sqrt: bool = False):
        """Fit data to a single polynomial with a given order."""
        x, y = self._x, self._y
        X = []
        if add_sqrt:
            X.append(np.sqrt(x))
        for power in np.arange(order, 0, -1, dtype=int):
            X.append(x**power)
        if intercept:
            X.append(np.ones(x.shape))
        X = np.array(X).T

        coeffs = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ coeffs
        y_rmse = rmse(y, y_pred)
        return (coeffs, y_pred, y_rmse), X

    @property
    def coeffs(self):
        """Return regression coefficients."""
        return self._coeffs

    @property
    def error(self):
        """Return error."""
        return self._error

    @property
    def best_model(self):
        """Return order and add_sqrt."""
        return (self._order, self._add_sqrt)
