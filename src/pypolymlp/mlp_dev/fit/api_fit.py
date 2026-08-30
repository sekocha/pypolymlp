"""API Class for estimating MLP coefficients."""

from typing import Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.fit.fit_cg import PolymlpFitCG
from pypolymlp.mlp_dev.fit.fit_learning_curve import PolymlpFitLearningCurve
from pypolymlp.mlp_dev.fit.fit_online_adam import PolymlpFitOnlineAdam
from pypolymlp.mlp_dev.fit.fit_standard import (
    PolymlpFitStandard,
    PolymlpFitStandardUseX,
)
from pypolymlp.mlp_dev.fit.fit_standard_loocv import (
    PolymlpFitStandardLOOCV,
    PolymlpFitStandardUseXLOOCV,
)


def fit_polymlp(
    params: PolymlpParams,
    train: DatasetList,
    test: Optional[DatasetList] = None,
    use_cg: bool = False,
    use_cv: bool = False,
    use_full_x: bool = False,
    batch_size: Optional[int] = None,
    gtol: float = 1e-2,
    max_iter: Optional[int] = None,
    error_threshold: float = 1e6,
    verbose: bool = False,
):
    """API function for estimating MLP coefficients."""
    if not use_cv and test is None:
        raise RuntimeError("Test data required.")

    if use_cg:
        if use_cv:
            raise RuntimeError("CV minimization not available for CG.")
        else:
            fitobj = PolymlpFitCG(
                params,
                train,
                test,
                gtol=gtol,
                max_iter=max_iter,
                verbose=verbose,
            )
    else:
        if use_cv:
            if use_full_x:
                fitobj = PolymlpFitStandardUseXLOOCV(
                    params,
                    train,
                    verbose=verbose,
                )
                # raise RuntimeError("CV minimization not available for use_full_x.")
            else:
                fitobj = PolymlpFitStandardLOOCV(
                    params,
                    train,
                    batch_size=batch_size,
                    verbose=verbose,
                )
        else:
            if use_full_x:
                fitobj = PolymlpFitStandardUseX(
                    params,
                    train,
                    test,
                    verbose=verbose,
                )
            else:
                fitobj = PolymlpFitStandard(
                    params,
                    train,
                    test,
                    batch_size=batch_size,
                    verbose=verbose,
                )

    fitobj.fit()

    rmses = np.array([mlp.rmse_test for mlp in fitobj.all_models])
    if np.all(rmses > error_threshold):
        raise RuntimeError("MLP estimation failed at all alpha values.")
    return fitobj


def fit_learning_curve(
    params: PolymlpParams,
    train: DatasetList,
    test: DatasetList,
    verbose: bool = False,
):
    """API function for estimating learning curve."""
    fitobj = PolymlpFitLearningCurve(
        params,
        train,
        test,
        verbose=verbose,
    )
    fitobj.fit()
    return fitobj


def fit_polymlp_online(
    params: PolymlpParams,
    train: DatasetList,
    coeffs: list | np.ndarray,
    beta: float = 0.95,
    batch_size: int = 100,
    gtol: float = 1e-2,
    n_epochs: int = 100,
    verbose: bool = False,
):
    """API function for updating MLP coefficients using online algorithms."""
    fitobj = PolymlpFitOnlineAdam(
        params,
        train,
        coeffs,
        beta=beta,
        batch_size=batch_size,
        gtol=gtol,
        n_epochs=n_epochs,
        verbose=verbose,
    )
    fitobj.fit()
    return fitobj
