"""API Class for estimating prediction errors."""

from typing import Literal, Optional

import numpy as np

from pypolymlp.core.dataset import DatasetList
from pypolymlp.mlp_dev.core.data_utils import PolymlpDataXY
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP

from .error_loocv import PolymlpErrorLOOCV, PolymlpErrorUseXLOOCV
from .error_rmse import PolymlpErrorRMSE


def eval_rmse(
    mlp: PolymlpDataMLP,
    datasets: DatasetList,
    stress_unit: Literal["eV", "GPa"] = "eV",
    log_energy: bool = True,
    log_force: bool = False,
    log_stress: bool = False,
    path_output: str = "./",
    tag: str = "train",
    verbose: bool = False,
):
    """Evaluate accuracy."""
    rmse_obj = PolymlpErrorRMSE(mlp, verbose=verbose)
    errors = rmse_obj.compute_error(
        datasets,
        stress_unit=stress_unit,
        log_energy=log_energy,
        log_force=log_force,
        log_stress=log_stress,
        path_output=path_output,
        tag=tag,
    )
    return errors


def compute_errors(
    mlp: PolymlpDataMLP,
    train: DatasetList,
    test: Optional[DatasetList] = None,
    inv_xtx: Optional[np.ndarray] = None,
    train_xy: Optional[PolymlpDataXY] = None,
    use_cv: bool = False,
    stress_unit: Literal["eV", "GPa"] = "eV",
    log_energy: bool = True,
    log_force: bool = False,
    log_stress: bool = False,
    path_output: str = "./",
    verbose: bool = False,
):
    """Compute errors for training (and test) datasets."""
    errors_train_obj = PolymlpErrorRMSE(mlp, verbose=verbose)
    _ = errors_train_obj.compute_error(
        train,
        stress_unit=stress_unit,
        log_energy=log_energy,
        log_force=log_force,
        log_stress=log_stress,
        path_output=path_output,
        tag="train",
    )
    if use_cv:
        if train_xy is None:
            if inv_xtx is None:
                raise RuntimeError("Inverse of X.T @ X required.")

            errors_test_obj = PolymlpErrorLOOCV(mlp, verbose=verbose)
            _ = errors_test_obj.compute_error(
                train,
                inv_xtx,
                stress_unit=stress_unit,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                path_output=path_output,
                tag="test",
            )
        else:
            if train_xy.inv_xtx is None:
                raise RuntimeError("Inverse of X.T @ X required.")

            errors_test_obj = PolymlpErrorUseXLOOCV(mlp, verbose=verbose)
            _ = errors_test_obj.compute_error(
                train,
                train_xy,
                stress_unit=stress_unit,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                path_output=path_output,
                tag="test",
            )
    else:
        if test is None:
            raise RuntimeError("Test dataset required.")
        errors_test_obj = PolymlpErrorRMSE(mlp, verbose=verbose)
        _ = errors_test_obj.compute_error(
            test,
            stress_unit=stress_unit,
            log_energy=log_energy,
            log_force=log_force,
            log_stress=log_stress,
            path_output=path_output,
            tag="test",
        )
    return (errors_train_obj, errors_test_obj)
