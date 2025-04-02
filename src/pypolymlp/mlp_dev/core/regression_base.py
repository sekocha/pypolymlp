"""Base class for regression"""

import gc
from abc import ABC, abstractmethod
from math import sqrt
from typing import Optional

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from pypolymlp.core.data_format import (
    PolymlpDataDFT,
    PolymlpDataMLP,
    PolymlpDataXY,
    PolymlpParams,
)
from pypolymlp.core.io_polymlp import save_mlp, save_mlps
from pypolymlp.core.io_polymlp_legacy import save_mlp_lammps, save_multiple_mlp_lammps
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase


class RegressionBase(ABC):
    """Base class for regression"""

    def __init__(
        self,
        polymlp_dev: PolymlpDevDataXYBase,
        train_xy: Optional[PolymlpDataXY] = None,
        test_xy: Optional[PolymlpDataXY] = None,
        verbose: bool = False,
    ):
        """Init method."""
        self._verbose = verbose
        self._params = polymlp_dev.params
        self._common_params = polymlp_dev.common_params
        self._hybrid = polymlp_dev.is_hybrid

        self._multiple_datasets = polymlp_dev.is_multiple_datasets

        if train_xy is None:
            self._train_xy = polymlp_dev.train_xy
        else:
            self._train_xy = train_xy

        if test_xy is None:
            self._test_xy = polymlp_dev.test_xy
        else:
            self._test_xy = test_xy

        self._train = polymlp_dev.train
        self._test = polymlp_dev.test
        self._scales = self._train_xy.scales
        self._best_model = None

        if self._hybrid:
            if self._train_xy is not None:
                self._cumulative_n_features = self._train_xy.cumulative_n_features
            elif self._test_xy is not None:
                self._cumulative_n_features = self._test_xy.cumulative_n_features

    @abstractmethod
    def fit(self):
        """Estimate regression coefficients."""
        pass

    def solve_linear_equation(self, A: np.ndarray, b: np.ndarray):
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
            if self._verbose:
                comment = "Error: The factorization could not be completed."
                print(" ", comment, flush=True)
            x = np.ones(x.shape) * 1e30
        return x

    def compute_inner_products(self, X=None, y=None, A=None, Xy=None):
        """Compute X.T @ X and X.T @ y."""
        if X is not None and y is not None:
            if self._verbose:
                print("Regression: computing inner products ...", flush=True)
            A = X.T @ X
            Xy = X.T @ y
        return A, Xy

    def rmse(self, true, pred):
        """Return RMSE."""
        return rmse(true, pred)

    def rmse_list(self, true, pred_list):
        """Return RMSE list."""
        return [rmse(true, p) for p in pred_list]

    def predict(self, coefs_array):
        """Compute rmse using X and y"""
        pred_train = (self._train_xy.x @ coefs_array).T
        pred_test = (self._test_xy.x @ coefs_array).T
        rmse_train = self.rmse_list(self._train_xy.y, pred_train)
        rmse_test = self.rmse_list(self._test_xy.y, pred_test)
        return pred_train, pred_test, rmse_train, rmse_test

    def predict_seq(self, coefs_array):
        """Compute rmse (train and test) using xtx, xty and y_sq"""
        rmse_train_array = self.predict_seq_train(coefs_array)
        rmse_test_array = self.predict_seq_test(coefs_array)
        return rmse_train_array, rmse_test_array

    def predict_seq_train(self, coefs_array):
        """Compute rmse (train) using xtx, xty and y_sq"""
        rmse_train_array = []
        for coefs in coefs_array.T:
            mse_train = self._compute_mse(
                self._train_xy.xtx,
                self._train_xy.xty,
                self._train_xy.y_sq_norm,
                self._train_xy.total_n_data,
                coefs,
            )
            try:
                rmse_train_array.append(sqrt(mse_train))
            except:
                rmse_train_array.append(0.0)
        return rmse_train_array

    def predict_seq_test(self, coefs_array):
        """Computing rmse (test) using xtx, xty and y_sq"""
        rmse_test_array = []
        for coefs in coefs_array.T:
            mse_test = self._compute_mse(
                self._test_xy.xtx,
                self._test_xy.xty,
                self._test_xy.y_sq_norm,
                self._test_xy.total_n_data,
                coefs,
            )
            try:
                rmse_test_array.append(sqrt(mse_test))
            except:
                rmse_test_array.append(1e10)
        return rmse_test_array

    def _compute_mse(self, xtx, xty, y_sq_norm, size, coefs):
        v1 = coefs @ (xtx @ coefs)
        v2 = -2 * coefs @ xty
        return (v1 + v2 + y_sq_norm) / size

    def save_mlp(self, filename="polymlp.yaml"):
        """Save polymlp.yaml files"""
        if self._hybrid == False:
            save_mlp(
                self._params,
                self._coeffs,
                self._scales,
                filename=filename,
            )
        else:
            save_mlps(
                self._params,
                self._cumulative_n_features,
                self._coeffs,
                self._scales,
                prefix=filename,
            )
        return self

    def save_mlp_lammps(self, filename="polymlp.lammps"):
        """Save polymlp.lammps files"""
        if self._hybrid == False:
            save_mlp_lammps(
                self._params,
                self._coeffs,
                self._scales,
                filename=filename,
            )
        else:
            save_multiple_mlp_lammps(
                self._params,
                self._cumulative_n_features,
                self._coeffs,
                self._scales,
                prefix=filename,
            )
        return self

    def hybrid_division(self, target):
        cumulative = self._cumulative_n_features
        list_target = []
        for i, params in enumerate(self._params):
            if i == 0:
                begin, end = 0, cumulative[0]
            else:
                begin, end = cumulative[i - 1], cumulative[i]
            list_target.append(np.array(target[begin:end]))
        return list_target

    @property
    def best_model(self):
        """
        Keys
        ----
        coeffs, scales, rmse, alpha, predictions (train, test)
        """
        return self._best_model

    @property
    def coeffs(self):
        if self._hybrid:
            return self.hybrid_division(self._coeffs)
        return self._coeffs

    @property
    def scales(self):
        if self._hybrid:
            return self.hybrid_division(self._scales)
        return self._scales

    @best_model.setter
    def best_model(self, model: PolymlpDataMLP):
        self._best_model = model
        self._coeffs = self._best_model.coeffs
        self._scales = self._best_model.scales

    @property
    def coeffs_vector(self):
        return self._coeffs

    @property
    def scales_vector(self):
        return self._scales

    @coeffs.setter
    def coeffs(self, array):
        self._coeffs = array
        self._best_model.coeffs = array

    @scales.setter
    def scales(self, array):
        self._scales = array
        self._best_model.scales = array

    @property
    def params(self) -> PolymlpParams:
        return self._params

    @property
    def train(self) -> PolymlpDataDFT:
        return self._train

    @property
    def test(self) -> PolymlpDataDFT:
        return self._test

    @property
    def train_xy(self) -> PolymlpDataXY:
        return self._train_xy

    @property
    def test_xy(self) -> PolymlpDataXY:
        return self._test_xy

    @train_xy.setter
    def train_xy(self, xy: PolymlpDataXY):
        self._train_xy = xy

    @test_xy.setter
    def test_xy(self, xy: PolymlpDataXY):
        self._test_xy = xy

    def delete_train_xy(self):
        del self._train_xy
        gc.collect()
        self._train_xy = None

    def delete_test_xy(self):
        del self._test_xy
        gc.collect()
        self._test_xy = None

    @property
    def is_multiple_datasets(self) -> bool:
        return self._multiple_datasets

    @property
    def is_hybrid(self) -> bool:
        return self._hybrid

    @property
    def verbose(self) -> bool:
        return self._verbose
