"""Base class for solving normal equations."""

from abc import ABC, abstractmethod

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore


class PolymlpFitBase(ABC):
    """Base class for solving normal equations."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        use_gradient: bool = False,
        verbose: bool = False,
    ):
        """Init method."""
        self._params = params
        self._train = train
        self._verbose = verbose

        self._polymlp = PolymlpDevCore(
            params,
            use_gradient=use_gradient,
            verbose=verbose,
        )

        self._best_model = None
        self._all_models = None

    @abstractmethod
    def fit(self):
        """Estimate polymlp coefficients."""
        pass

    @property
    def best_model(self):
        """Return best model."""
        return self._best_model

    @property
    def all_models(self):
        """Return all models."""
        return self._all_models
