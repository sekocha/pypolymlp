"""Functions for estimating regression coefficients from datasets."""

from typing import Optional

from pypolymlp.core.utils import check_memory_size_in_regression
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.utils_sequential import get_auto_batch_size
from pypolymlp.mlp_dev.standard.learning_curve import LearningCurve
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression


def fit(
    polymlp_in: PolymlpDevData,
    batch_size: Optional[int] = None,
    verbose: bool = False,
):
    """Estimate MLP coefficients without computing entire X.

    Parameters
    ----------
    polymlp_in: PolymlpDevData instance.
    batch_size: Batch size for sequential regression.
                If None, the batch size is automatically determined
                depending on the memory size and number of features.
    """
    if polymlp_in.train is None or polymlp_in.test is None:
        raise RuntimeError("Datasets not found.")

    if batch_size is None:
        batch_size = get_auto_batch_size(polymlp_in.n_features, verbose=verbose)

    mem_req = check_memory_size_in_regression(polymlp_in.n_features)
    if verbose:
        print("Minimum memory required for solver in GB:", mem_req, flush=True)
        print("Memory required for allocating X additionally.", flush=True)
        print("Batch size for computing X:", batch_size, flush=True)

    polymlp = PolymlpDevDataXYSequential(polymlp_in, verbose=verbose)
    polymlp.run_train(batch_size=batch_size)

    reg = Regression(polymlp, verbose=verbose).fit(
        seq=True,
        clear_data=True,
        batch_size=batch_size,
    )
    return reg


def fit_standard(polymlp_in: PolymlpDevData, verbose: bool = False):
    """Estimate MLP coefficients with direct evaluation of X."""
    if polymlp_in.train is None or polymlp_in.test is None:
        raise RuntimeError("Datasets not found.")

    polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
    if verbose:
        polymlp.print_data_shape()
    reg = Regression(polymlp, verbose=verbose).fit(seq=False)
    return reg


def fit_learning_curve(polymlp_in: PolymlpDevData, verbose: bool = False):
    """Calculate learning curve."""
    if polymlp_in.train is None or polymlp_in.test is None:
        raise RuntimeError("Datasets not found.")

    if len(polymlp_in.train) > 1:
        raise RuntimeError("Use single dataset for learning curve calculation")

    polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
    total_n_atoms = polymlp._train[0].total_n_atoms

    learning = LearningCurve(polymlp, total_n_atoms, verbose=verbose)
    learning.run()
    return learning
