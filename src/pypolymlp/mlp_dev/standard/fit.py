"""Functions for estimating regression coefficients from datasets."""

from typing import Optional, Union

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.mlpdev import PolymlpDevCore

# from pypolymlp.mlp_dev.standard.learning_curve import LearningCurve
# from pypolymlp.mlp_dev.standard.regression import Regression


def fit(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients without computing entire X.

    Parameters
    ----------
    batch_size: Batch size for sequential regression.
                If None, the batch size is automatically determined
                depending on the memory size and number of features.
    """
    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xtx_xty(train, batch_size=batch_size)
    scales = train_xy.scales
    min_energy = train_xy.min_energy

    train_xy.clear_data()
    test_xy = polymlp.calc_xtx_xty(
        test,
        scales=scales,
        min_energy=min_energy,
        batch_size=batch_size,
    )
    test_xy.clear_data()

    return None
    # polymlp.run_train(batch_size=batch_size)

    # reg = Regression(polymlp, verbose=verbose).fit(
    #     seq=True,
    #     clear_data=True,
    #     batch_size=batch_size,
    # )


#    return reg


def fit_standard(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    verbose: bool = False,
):
    """Estimate MLP coefficients with direct evaluation of X."""

    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    train_xy = polymlp.calc_xy(train)
    train_xy.clear_data()
    test_xy = polymlp.calc_xy(
        test,
        scales=train_xy.scales,
        min_energy=train_xy.min_energy,
    )
    test_xy.clear_data()

    # def fit_standard(polymlp_in: PolymlpDevData, verbose: bool = False):
    #     """Estimate MLP coefficients with direct evaluation of X."""
    #     if polymlp_in.train is None or polymlp_in.test is None:
    #         raise RuntimeError("Datasets not found.")
    #
    #     polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
    #     if verbose:
    #         polymlp.print_data_shape()
    #     reg = Regression(polymlp, verbose=verbose).fit(seq=False)
    #     return reg
    return None


def fit_learning_curve(
    params: Union[PolymlpParams, list[PolymlpParams]],
    train: list[PolymlpDataDFT],
    test: list[PolymlpDataDFT],
    verbose: bool = False,
):
    """Calculate learning curve."""

    polymlp = PolymlpDevCore(params)
    polymlp.check_memory_size_in_regression()

    if len(train) > 1:
        raise RuntimeError("Use single dataset for learning curve calculation")

    return None


#     polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
#     total_n_atoms = polymlp._train[0].total_n_atoms
#
#     learning = LearningCurve(polymlp, total_n_atoms, verbose=verbose)
#     learning.run()
#     return learning
