"""Classes for estimating regression coefficients from datasets."""

from typing import Optional

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams

from .fit_base import PolymlpFitBase
from .solvers_standard import solver_ridge


class PolymlpFitStandard(PolymlpFitBase):
    """Class for estimating MLP coefficients without computing entire X."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        test: DatasetList,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train: Training datasets.
        test: Test datasets.
        batch_size: Batch size for sequential regression.
                    If None, the batch size is automatically determined
                    depending on the memory size and number of features.
        """
        super().__init__(params, train, verbose=verbose)

        self._test = test
        self._batch_size = batch_size

    def fit(self):
        """Estimate MLP coefficients."""
        self._polymlp.check_memory_size_in_regression()

        train_xy = self._polymlp.calc_xtx_xty(self._train, batch_size=self._batch_size)
        coefs = solver_ridge(
            xtx=train_xy.xtx,
            xty=train_xy.xty,
            alphas=self._params.alphas,
            verbose=self._verbose,
        )

        rmse_train = self._polymlp.compute_rmse(coefs, train_xy, check_singular=True)
        train_xy.clear_data()

        test_xy = self._polymlp.calc_xtx_xty(
            self._test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
            batch_size=self._batch_size,
        )
        rmse_test = self._polymlp.compute_rmse(coefs, test_xy)
        test_xy.clear_data()

        if self._verbose:
            self._polymlp.print_model_selection_log(rmse_train, rmse_test)

        self._best_model = self._polymlp.get_best_model(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        self._all_models = self._polymlp.get_all_models(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )

        return self


class PolymlpFitStandardUseX(PolymlpFitBase):
    """Class for estimating MLP coefficients with direct evaluation of X."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        test: DatasetList,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train: Training datasets.
        test: Test datasets.
        """
        super().__init__(params, train, verbose=verbose)

        self._test = test

    def fit(self):
        """Estimate MLP coefficients."""
        self._polymlp.check_memory_size_in_regression()

        train_xy = self._polymlp.calc_xy(self._train)
        coefs = solver_ridge(
            x=train_xy.x,
            y=train_xy.y,
            alphas=self._params.alphas,
            verbose=self._verbose,
        )
        rmse_train = self._polymlp.compute_rmse(coefs, train_xy, check_singular=True)
        train_xy.clear_data()

        test_xy = self._polymlp.calc_xy(
            self._test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
        )
        rmse_test = self._polymlp.compute_rmse(coefs, test_xy)
        test_xy.clear_data()

        self._best_model = self._polymlp.get_best_model(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        self._all_models = self._polymlp.get_all_models(
            coefs,
            train_xy.scales,
            rmse_train,
            rmse_test,
            train_xy.cumulative_n_features,
        )
        if self._verbose:
            self._polymlp.print_model_selection_log(rmse_train, rmse_test)
        return self
