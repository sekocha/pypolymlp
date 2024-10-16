"""Base class for PolymlpDevDataXY."""

import gc
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpDataXY
from pypolymlp.mlp_dev.core.features import Features, FeaturesHybrid
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.utils_weights import apply_weight_percentage


class PolymlpDevDataXYBase(ABC):
    """Base class for PolymlpDevDataXY."""

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose: bool = True):
        self._params = polymlp_dev_data.params
        self._common_params = polymlp_dev_data.common_params
        self._verbose = verbose

        self._train = polymlp_dev_data.train
        self._test = polymlp_dev_data.test
        self._min_energy = polymlp_dev_data.min_energy

        self._multiple_datasets = polymlp_dev_data.is_multiple_datasets
        self._hybrid = polymlp_dev_data.is_hybrid

        if self.is_hybrid == False:
            self._feature_class = Features
        else:
            self._feature_class = FeaturesHybrid

        self._train_xy = None
        self._test_xy = None
        self._scales = None

    def print_data_shape(self):

        x = self._train_xy.x
        ne, nf, ns = self._train_xy.n_data
        print("Training Dataset:", x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)

        x = self._test_xy.x
        ne, nf, ns = self._test_xy.n_data
        print("Test Dataset:", x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)
        return self

    def apply_scales(self):
        if self._train_xy is None:
            raise ValueError("Not found: PolymlpDataXY.")

        x = self._train_xy.x
        ne, nf, ns = self._train_xy.n_data
        self._scales = np.std(x[:ne], axis=0)
        self._scales[np.abs(self._scales) < 1e-30] = 1.0

        self._train_xy.x /= self._scales
        self._test_xy.x /= self._scales

        self._train_xy.scales = self._scales
        self._test_xy.scales = self._scales

        return self

    def apply_weights(self, weight_stress=0.1):
        if self._train_xy is None:
            raise ValueError("Not found: PolymlpDataXY")

        self._train_xy = self._apply_weights_single_set(
            self._train,
            self._train_xy,
            weight_stress=weight_stress,
        )
        self._test_xy = self._apply_weights_single_set(
            self._test,
            self._test_xy,
            weight_stress=weight_stress,
        )

    def _apply_weights_single_set(
        self,
        data_dft: Union[PolymlpDataDFT, list[PolymlpDataDFT]],
        data_xy: PolymlpDataXY,
        weight_stress: float = 0.1,
    ) -> PolymlpDataXY:

        first_indices = data_xy.first_indices
        x = data_xy.x
        n_data, n_features = x.shape
        y = np.zeros(n_data)
        w = np.ones(n_data)

        if self._multiple_datasets == False:
            indices = first_indices[0]
            x, y, w = apply_weight_percentage(
                x,
                y,
                w,
                data_dft,
                self._common_params,
                indices,
                weight_stress=weight_stress,
                min_e=self._min_energy,
            )
        else:
            for dft, indices in zip(data_dft, first_indices):
                x, y, w = apply_weight_percentage(
                    x,
                    y,
                    w,
                    dft,
                    self._common_params,
                    indices,
                    weight_stress=weight_stress,
                    min_e=self._min_energy,
                )

        data_xy.x = x
        data_xy.y = y
        data_xy.weight = w
        data_xy.scales = self._scales
        return data_xy

    @abstractmethod
    def run(self):
        pass

    @property
    def params(self):
        return self._params

    @property
    def common_params(self):
        return self._common_params

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
    def train_xy(self, data: PolymlpDataXY):
        self._train_xy = data

    @test_xy.setter
    def test_xy(self, data: PolymlpDataXY):
        self._test_xy = data

    def delete_train_xy(self):
        del self._train_xy
        gc.collect()

    def delete_test_xy(self):
        del self._test_xy
        gc.collect()

    @property
    def features_class(self):
        return self._feature_class

    @property
    def is_multiple_datasets(self):
        return self._multiple_datasets

    @property
    def is_hybrid(self):
        return self._hybrid

    @property
    def min_energy(self):
        return self._min_energy

    @property
    def verbose(self):
        return self._verbose
