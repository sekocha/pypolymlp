"""Base class for PolymlpDevDataXY."""

import gc
from abc import ABC, abstractmethod

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpDataXY
from pypolymlp.mlp_dev.core.features import Features, FeaturesHybrid
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData


class PolymlpDevDataXYBase(ABC):
    """Base class for PolymlpDevDataXY."""

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose: bool = False):
        """Init method."""
        self._params = polymlp_dev_data.params
        self._common_params = polymlp_dev_data.common_params
        self._verbose = verbose

        self._train = polymlp_dev_data.train
        self._test = polymlp_dev_data.test
        self._check_multiple_dataset_format()
        self._min_energy = polymlp_dev_data.min_energy

        self._hybrid = polymlp_dev_data.is_hybrid
        if not self._hybrid:
            self._feature_class = Features
        else:
            self._feature_class = FeaturesHybrid

        self._train_xy = None
        self._test_xy = None
        self._scales = None

    def _check_multiple_dataset_format(self):
        """Check whether dataset is given in multiple dataset format."""
        if not isinstance(self._train, (list, tuple, np.ndarray)):
            raise RuntimeError(
                "Training dataset is not given in multiple dataset format."
            )
        if not isinstance(self._test, (list, tuple, np.ndarray)):
            raise RuntimeError("Test dataset is not given in multiple dataset format.")
        self._multiple_datasets = True
        return self

    @abstractmethod
    def run(self):
        """Compute (X, y) or (X.T @ X, X.T @ y)."""
        pass

    @property
    def params(self):
        """Return parameters for developing polymlp."""
        return self._params

    @property
    def common_params(self):
        """Return common parameters in hybrid models for developing polymlp."""
        return self._common_params

    @property
    def train(self) -> list[PolymlpDataDFT]:
        """Return training DFT datasets."""
        return self._train

    @property
    def test(self) -> list[PolymlpDataDFT]:
        """Return test DFT datasets."""
        return self._test

    @property
    def train_xy(self) -> PolymlpDataXY:
        """Return training (X, y) or (X.T @ X, X.T @ y)."""
        return self._train_xy

    @property
    def test_xy(self) -> PolymlpDataXY:
        """Return test (X, y) or (X.T @ X, X.T @ y)."""
        return self._test_xy

    @train_xy.setter
    def train_xy(self, data: PolymlpDataXY):
        """Set training (X, y) or (X.T @ X, X.T @ y)."""
        self._train_xy = data

    @test_xy.setter
    def test_xy(self, data: PolymlpDataXY):
        """Set test (X, y) or (X.T @ X, X.T @ y)."""
        self._test_xy = data

    def delete_train_xy(self):
        """Delete training (X, y) or (X.T @ X, X.T @ y)."""
        del self._train_xy
        gc.collect()

    def delete_test_xy(self):
        """Delete test (X, y) or (X.T @ X, X.T @ y)."""
        del self._test_xy
        gc.collect()

    @property
    def scales(self):
        """Return scales of X.."""
        return self._scales

    @property
    def features_class(self):
        """Return class for computing features."""
        return self._feature_class

    @property
    def is_multiple_datasets(self):
        """Return whether dataset is givn in multiple dataset format."""
        return self._multiple_datasets

    @property
    def is_hybrid(self):
        """Return whether hybrid model is used or not."""
        return self._hybrid

    @property
    def min_energy(self):
        """Return minimum energy in dataset entries."""
        return self._min_energy

    @property
    def verbose(self):
        """Return verbose."""
        return self._verbose
