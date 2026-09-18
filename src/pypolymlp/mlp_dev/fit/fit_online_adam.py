"""Class for estimating MLP coefficients using online Adam."""

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams

from .fit_base import PolymlpFitBase
from .solvers_adam import solver_adam


class PolymlpFitOnlineAdam(PolymlpFitBase):
    """Class for estimating MLP coefficients using online Adam."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        coeffs: list | np.ndarray,
        gtol: float = 1e-2,
        n_epochs: int = 100,
        beta: float = 0.95,
        batch_size: int = 100,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train:  Training datasets.
        coeffs: Initial scaled coefficients of polynomial MLP.
        """
        super().__init__(params, train, use_gradient=True, verbose=verbose)

        self._coef0 = coeffs
        self._gtol = gtol
        self._n_epochs = n_epochs
        self._beta = beta
        self._batch_size = batch_size

    def fit(self):
        """Estimate MLP coefficients."""
        train_xy = self._polymlp.calc_xy(self._train)
        train_xy.x *= train_xy.scales
        coeffs = solver_adam(
            x=train_xy.x,
            y=train_xy.y,
            coef0=self._coef0,
            beta=self._beta,
            batch_size=self._batch_size,
            gtol=self._gtol,
            n_epochs=self._n_epochs,
            verbose=self._verbose,
        )
        rmse_train = self._polymlp.compute_rmse(
            coeffs.reshape((-1, 1)), train_xy, check_singular=True
        )[0]
        train_xy.clear_data()

        self._best_model = self._polymlp.set_model(
            coeffs,
            np.ones(coeffs.shape),
            rmse_train,
            rmse_test=None,
            cumulative_n_features=train_xy.cumulative_n_features,
        )
        return self
