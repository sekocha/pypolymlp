"""Utility functions for fitting thermodynamic properties."""

import itertools
from typing import Optional

import numpy as np
import scipy

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
        self._x = np.array(x)
        self._y = np.array(y)
        self._coeffs = None
        self._error = None
        self._order = None
        self._add_sqrt = None

    def eval(self, x: float):
        """Evaluate value of fitted polynomial at given x."""
        coeffs = self._coeffs[1:] if self._add_sqrt else self._coeffs
        val = np.polyval(coeffs, x)
        if self._add_sqrt:
            val += self._coeffs[0] * np.power(x, 0.5)
        return val

    def eval_derivative(self, x: float):
        """Evaluate derivative of fitted polynomial at given x."""
        coeffs = self._coeffs[1:] if self._add_sqrt else self._coeffs
        deriv = coeffs * np.arange(len(coeffs) - 1, -1, -1, dtype=int)
        deriv = deriv[:-1]
        val = np.polyval(deriv, x)
        if self._add_sqrt:
            ids = np.where(np.abs(x) < 1e-10)[0]
            x[ids] = 1.0
            val += 0.5 * self._coeffs[0] * np.power(x, -0.5)
            x[ids] = 0.0
        return val

    def fit(
        self,
        order: Optional[int] = None,
        max_order: int = 4,
        intercept: bool = True,
        first_order: bool = True,
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
                res = self._fit_single(
                    order,
                    intercept=intercept,
                    first_order=first_order,
                    add_sqrt=add_sqrt,
                )
                (_, y_pred, y_rmse), X = res
                cv = loocv(X, self._y, y_pred)
                if min_loocv > cv:
                    min_loocv = cv
                    best_order, best_add_sqrt = order, add_sqrt

        (coeffs, y_pred, y_rmse), _ = self._fit_single(
            best_order,
            intercept=intercept,
            first_order=first_order,
            add_sqrt=best_add_sqrt,
        )
        self._coeffs = self._adjust_coeffs(coeffs, intercept, first_order)
        self._error = y_rmse
        self._order = best_order
        self._add_sqrt = best_add_sqrt
        return self

    def _adjust_coeffs(self, coeffs: np.ndarray, intercept: bool, first_order: bool):
        """Adjust regression coefficients."""
        coeffs = list(coeffs)
        if intercept:
            intercept_c = coeffs[-1]
            poly_c = coeffs[:-1]
        else:
            intercept_c = 0.0
            poly_c = coeffs

        if not first_order:
            poly_c.append(0.0)

        coeffs = poly_c
        coeffs.append(intercept_c)
        return np.array(coeffs)

    def _fit_single(
        self,
        order: int,
        intercept: bool = True,
        first_order: bool = True,
        add_sqrt: bool = False,
    ):
        """Fit data to a single polynomial with a given order."""
        x, y = self._x, self._y
        X = []
        range_end = 0 if first_order else 1
        if add_sqrt:
            X.append(np.sqrt(x))
        for power in np.arange(order, range_end, -1, dtype=int):
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


def fit_cv_temperature(temperatures: np.ndarray, cv: np.ndarray, verbose: bool = True):
    """Fit Cv-temperature data to a polynomial."""
    polyfit = Polyfit(temperatures, cv)
    polyfit.fit(max_order=6, intercept=False, first_order=False, add_sqrt=True)
    cv_pred = polyfit.eval(temperatures)
    if verbose:
        print("Cv-Temperature Fit", flush=True)
        print("  RMSE:  ", polyfit.error, flush=True)
        print("  model: ", polyfit.best_model, flush=True)
        print("  # temp., Cv(observed), Cv(fitted):", flush=True)
        for t, cv1, cv2 in zip(temperatures, cv, cv_pred):
            print("   ", t, np.round(cv1, 5), np.round(cv2, 5), flush=True)
    return cv_pred


def _func_poly(x, *args):
    """Define polynomial function."""
    return np.polyval(args[0], x)


def _fit_poly(
    f1: np.ndarray,
    f2: np.ndarray,
    order: Optional[int] = None,
    max_order: int = 6,
):
    """Fit two data using a polynomial."""
    p1 = Polyfit(f1[:, 0], f1[:, 1]).fit(order=order, max_order=max_order)
    p2 = Polyfit(f2[:, 0], f2[:, 1]).fit(order=order, max_order=max_order)

    z1, z2 = p1.coeffs, p2.coeffs
    len_diff = len(z1) - len(z2)
    if len_diff > 0:
        z2 = np.hstack([np.zeros(len_diff), z2])
    elif len_diff < 0:
        z1 = np.hstack([np.zeros(-len_diff), z1])
    return z1, z2


def fit_solve_poly(
    f1: np.ndarray,
    f2: np.ndarray,
    f0: float = 0.0,
    order: Optional[int] = None,
    max_order: int = 6,
):
    """Fit and solve delta f = 0."""
    z1, z2 = _fit_poly(f1, f2, order=order, max_order=max_order)
    coeffs = z1 - z2
    res = scipy.optimize.fsolve(_func_poly, f0, args=coeffs)
    return res[0]


def _func_spline(x, *args):
    """Define spline function."""
    sp1, sp2 = args
    return sp1(x) - sp2(x)


def fit_solve_spline(f1: np.ndarray, f2: np.ndarray, f0: float = 0.0, k: int = 3):
    """Fit and solve delta f = 0."""
    sp1 = scipy.interpolate.make_interp_spline(f1[:, 0], f1[:, 1], k=k)
    sp2 = scipy.interpolate.make_interp_spline(f2[:, 0], f2[:, 1], k=k)
    args = sp1, sp2
    res = scipy.optimize.fsolve(_func_spline, f0, args=args)
    return res[0]
