"""Tests of PolymlpEvalAccuracy."""

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.errors.api_errors import compute_errors, eval_rmse
from pypolymlp.mlp_dev.fit.fit_standard_loocv import PolymlpFitStandardUseXLOOCV

cwd = Path(__file__).parent


def test_compute_errors(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    train = copy.deepcopy(datasets)
    test = copy.deepcopy(datasets)
    obj, _ = compute_errors(
        mlp_mp_149, train, test=test, use_cv=False, log_energy=False
    )
    tag = "Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(5.84708583252534e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.0028245295894005276)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(4.668251936400402e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.002250798354169759)

    _, obj = compute_errors(
        mlp_mp_149,
        train,
        test=test,
        use_cv=False,
        log_energy=True,
        log_force=True,
        log_stress=False,
        path_output="tmp",
    )
    shutil.rmtree("tmp")


def test_compute_cv(regdata_mp_149, mlp_mp_149, dataxy_xtx_xty_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    data = copy.deepcopy(datasets)
    xtx = dataxy_xtx_xty_mp_149.xtx
    inv_xtx = np.linalg.inv(xtx)

    _, obj = compute_errors(
        mlp_mp_149, data, inv_xtx=inv_xtx, use_cv=True, log_energy=False
    )
    tag = "LOOCV:Train_Data_from_files"
    # TODO: CV values must be the same as the usex version.
    assert obj.errors[tag]["energy"] == pytest.approx(1.8956990735382943e-12)
    assert obj.errors[tag]["force"] == pytest.approx(0.05383648494520971)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(1.5134410137908146e-12)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.003709359247952824)

    _, obj = compute_errors(
        mlp_mp_149,
        data,
        inv_xtx=inv_xtx,
        use_cv=True,
        log_energy=True,
        log_force=True,
        log_stress=False,
        path_output="tmp",
    )
    shutil.rmtree("tmp")


def test_compute_cv_usex(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    params, datasets = regdata_mp_149
    data = copy.deepcopy(datasets)

    fit = PolymlpFitStandardUseXLOOCV(params, datasets)
    fit.fit()
    train_xy = fit.train_xy

    _, obj = compute_errors(
        mlp_mp_149, data, train_xy=train_xy, use_cv=True, log_energy=False
    )
    tag = "LOOCV:Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(5.880177980263584e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.002836519189731207)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(4.694669298619145e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.002260251564372544)

    _, obj = compute_errors(
        mlp_mp_149,
        data,
        train_xy=train_xy,
        use_cv=True,
        log_energy=True,
        log_force=True,
        log_stress=False,
        path_output="tmp",
    )
    shutil.rmtree("tmp")


def test_eval_rmse(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    data = datasets
    errors = eval_rmse(mlp_mp_149, data, log_energy=False)
    tag = "Train_Data_from_files"
    assert errors[tag]["energy"] == pytest.approx(5.84708583252534e-06)
    assert errors[tag]["force"] == pytest.approx(0.0028245295894005276)
    assert errors[tag]["energy_mae"] == pytest.approx(4.668251936400402e-06)
    assert errors[tag]["force_mae"] == pytest.approx(0.002250798354169759)


def test_class_rmse(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    train = copy.deepcopy(datasets)
    test = copy.deepcopy(datasets)
    obj, _ = compute_errors(
        mlp_mp_149, train, test=test, use_cv=False, log_energy=False
    )
    for k, v in obj.errors.items():
        obj.print_error(v)
    obj.write_error_yaml(filename="tmp.yaml")
    os.remove("tmp.yaml")
