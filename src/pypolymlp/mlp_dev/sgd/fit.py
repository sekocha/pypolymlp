"""Functions sgd for estimating regression coefficients from datasets."""

# from typing import Optional, Union
#
# from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
#
# # from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore, eval_accuracy
# from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore
# from pypolymlp.mlp_dev.sgd.solvers import solver_sgd
#

# def fit_sgd(
#     params: Union[PolymlpParams, list[PolymlpParams]],
#     common_params: PolymlpParams,
#     train: list[PolymlpDataDFT],
#     test: list[PolymlpDataDFT],
#     batch_size: Optional[int] = None,
#     verbose: bool = False,
# ):
#     """Estimate MLP coefficients without computing entire X.
#
#     Parameters
#     ----------
#     batch_size: Batch size for sequential regression.
#                 If None, the batch size is automatically determined
#                 depending on the memory size and number of features.
#     """
#     polymlp = PolymlpDevCore(params)
#     polymlp.check_memory_size_in_regression()
#
#     train_xy = polymlp.calc_xy(train)
#
#     coefs = solver_sgd()
#
#     # rmse_train = compute_rmse(coefs, train_xy, check_singular=True)
#     # train_xy.clear_data()
#
#     # test_xy = polymlp.calc_xy(
#     #     test,
#     #     scales=train_xy.scales,
#     #     min_energy=train_xy.min_energy,
#     # )
#     # rmse_test = compute_rmse(coefs, test_xy)
#     # test_xy.clear_data()
#
#     # return best_model
#     return None
