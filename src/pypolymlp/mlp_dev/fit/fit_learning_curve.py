"""Class for calculating learning curve."""

# from typing import Optional
#
# from pypolymlp.core.dataset import DatasetList
# from pypolymlp.core.params import PolymlpParams
# from pypolymlp.mlp_dev.core.api_mlpdev import PolymlpDevCore, eval_accuracy
#
# from .fit_base import PolymlpFitBase
# from .solvers_standard import solver_ridge
#
#
# class PolymlpFitLearningCurve(PolymlpFitBase):
#     """Class for calculating learning curve."""
#
#     def __init__(
#         self,
#         params: PolymlpParams,
#         train: DatasetList,
#         test: DatasetList,
#         verbose: bool = False,
#     ):
#         """Init method.
#
#         params: Parameters of polymlp.
#         train: Training datasets.
#         test: Test datasets.
#         """
#         super().__init__(params, train, verbose=verbose)
#
#         self._test = test
#
#     def fit(self):
#         """Estimate learning curve."""
#         self._polymlp.check_memory_size_in_regression()
#
#         train_xy = self._polymlp.calc_xtx_xty(self._train, batch_size=self._batch_size)
#         coefs = solver_ridge(
#             xtx=train_xy.xtx,
#             xty=train_xy.xty,
#             alphas=self._params.alphas,
#             verbose=self._verbose,
#         )
#
#         rmse_train = self._polymlp.compute_rmse(coefs, train_xy, check_singular=True)
#         train_xy.clear_data()
#
#         test_xy = self._polymlp.calc_xtx_xty(
#             self._test,
#             scales=train_xy.scales,
#             min_energy=train_xy.min_energy,
#             batch_size=self._batch_size,
#         )
#         rmse_test = self._polymlp.compute_rmse(coefs, test_xy)
#         test_xy.clear_data()
#
#         if self._verbose:
#             self._polymlp.print_model_selection_log(rmse_train, rmse_test)
#
#         self._best_model = self._polymlp.get_best_model(
#             coefs,
#             train_xy.scales,
#             rmse_train,
#             rmse_test,
#             train_xy.cumulative_n_features,
#         )
#         return self
#
#

# def fit_learning_curve(
#     params: PolymlpParams,
#     train: DatasetList,
#     test: DatasetList,
#     verbose: bool = False,
# ):
#     """Calculate learning curve.
#
#     Parameters
#     ----------
#     params: Parameters of polymlp.
#     train: Training datasets.
#     test: Test datasets.
#     """
#     if len(train) != 1:
#         raise RuntimeError(
#             "Number of training datasets must be one for learning curve."
#         )
#
#     polymlp = PolymlpDevCore(params, verbose=verbose)
#     polymlp.check_memory_size_in_regression()
#
#     train_xy = polymlp.calc_xy(train)
#     test_xy = polymlp.calc_xy(
#         test,
#         scales=train_xy.scales,
#         min_energy=train_xy.min_energy,
#     )
#
#     if verbose:
#         print("Calculate learning curve.", flush=True)
#
#     error_log = []
#     n_train = train_xy.n_structures
#     for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
#         if verbose:
#             print("------------- n_samples:", n_samples, "-------------", flush=True)
#
#         x, y = train_xy.slice(n_samples, train[0].total_n_atoms)
#         coefs = solver_ridge(
#             x=x,
#             y=y,
#             alphas=params.alphas,
#             verbose=False,
#         )
#         rmse_train = polymlp.compute_rmse(coefs, x=x, y=y)
#         rmse_test = polymlp.compute_rmse(coefs, test_xy)
#         best_model = polymlp.get_best_model(
#             coefs,
#             train_xy.scales,
#             rmse_train,
#             rmse_test,
#             train_xy.cumulative_n_features,
#         )
#         if verbose:
#             polymlp.print_model_selection_log(rmse_train, rmse_test)
#
#         error = eval_accuracy(best_model, test, log_energy=False, tag="test")
#         for val in error.values():
#             error_log.append([n_samples, val])
#
#     if verbose:
#         print_learning_curve_log(error_log)
#
#     return error_log
