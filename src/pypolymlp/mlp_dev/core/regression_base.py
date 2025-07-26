"""Base class for regression"""

# import gc
# from abc import ABC, abstractmethod
# from math import sqrt
# from typing import Optional
#
# import numpy as np
# from scipy.linalg.lapack import get_lapack_funcs
#
# from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpDataXY, PolymlpParams
# from pypolymlp.core.io_polymlp import save_mlp, save_mlps
# from pypolymlp.core.io_polymlp_legacy import save_mlp_lammps, save_multiple_mlp_lammps
# from pypolymlp.core.utils import rmse
# from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP
# from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase
#
#
# class RegressionBase(ABC):
#     """Base class for regression"""
#
#     def save_mlp(self, filename="polymlp.yaml"):
#         """Save polymlp.yaml files"""
#         if self._hybrid == False:
#             save_mlp(
#                 self._params,
#                 self._coeffs,
#                 self._scales,
#                 filename=filename,
#             )
#         else:
#             save_mlps(
#                 self._params,
#                 self._cumulative_n_features,
#                 self._coeffs,
#                 self._scales,
#                 prefix=filename,
#             )
#         return self
#
#     def save_mlp_lammps(self, filename="polymlp.lammps"):
#         """Save polymlp.lammps files"""
#         if self._hybrid == False:
#             save_mlp_lammps(
#                 self._params,
#                 self._coeffs,
#                 self._scales,
#                 filename=filename,
#             )
#         else:
#             save_multiple_mlp_lammps(
#                 self._params,
#                 self._cumulative_n_features,
#                 self._coeffs,
#                 self._scales,
#                 prefix=filename,
#             )
#         return self
#
#     def hybrid_division(self, target):
#         cumulative = self._cumulative_n_features
#         list_target = []
#         for i, params in enumerate(self._params):
#             if i == 0:
#                 begin, end = 0, cumulative[0]
#             else:
#                 begin, end = cumulative[i - 1], cumulative[i]
#             list_target.append(np.array(target[begin:end]))
#         return list_target
#
#     @property
#     def best_model(self):
#         """
#         Keys
#         ----
#         coeffs, scales, rmse, alpha, predictions (train, test)
#         """
#         return self._best_model
#
#     @property
#     def coeffs(self):
#         if self._hybrid:
#             return self.hybrid_division(self._coeffs)
#         return self._coeffs
#
#     @property
#     def scales(self):
#         if self._hybrid:
#             return self.hybrid_division(self._scales)
#         return self._scales
#
#     @best_model.setter
#     def best_model(self, model: PolymlpDataMLP):
#         self._best_model = model
#         self._coeffs = self._best_model.coeffs
#         self._scales = self._best_model.scales
#
#     @property
#     def coeffs_vector(self):
#         return self._coeffs
#
#     @property
#     def scales_vector(self):
#         return self._scales
#
#     @coeffs.setter
#     def coeffs(self, array):
#         self._coeffs = array
#         self._best_model.coeffs = array
#
#     @scales.setter
#     def scales(self, array):
#         self._scales = array
#         self._best_model.scales = array
#
#     @property
#     def params(self) -> PolymlpParams:
#         return self._params
#
#     @property
#     def train(self) -> PolymlpDataDFT:
#         return self._train
#
#     @property
#     def test(self) -> PolymlpDataDFT:
#         return self._test
#
#     @property
#     def train_xy(self) -> PolymlpDataXY:
#         return self._train_xy
#
#     @property
#     def test_xy(self) -> PolymlpDataXY:
#         return self._test_xy
#
#     @train_xy.setter
#     def train_xy(self, xy: PolymlpDataXY):
#         self._train_xy = xy
#
#     @test_xy.setter
#     def test_xy(self, xy: PolymlpDataXY):
#         self._test_xy = xy
#
#     def delete_train_xy(self):
#         del self._train_xy
#         gc.collect()
#         self._train_xy = None
#
#     def delete_test_xy(self):
#         del self._test_xy
#         gc.collect()
#         self._test_xy = None
#
#     @property
#     def is_multiple_datasets(self) -> bool:
#         return self._multiple_datasets
#
#     @property
#     def is_hybrid(self) -> bool:
#         return self._hybrid
#
#     @property
#     def verbose(self) -> bool:
#         return self._verbose
