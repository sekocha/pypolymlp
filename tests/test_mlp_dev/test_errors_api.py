"""Tests of PolymlpEvalAccuracy."""

import copy
import os
import shutil
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.errors.api_errors import compute_errors, eval_rmse
from pypolymlp.mlp_dev.fit.fit_standard_loocv import (
    PolymlpFitStandardLOOCV,
    PolymlpFitStandardUseXLOOCV,
)

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
    assert obj.errors[tag]["energy"] == pytest.approx(1.7723796993157736e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.0008471005821760739)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(1.381235266206815e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.0006750500787842289)

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


def test_compute_cv(regdata_mp_149):
    """Test for compute_errors."""
    params, datasets = regdata_mp_149
    data = copy.deepcopy(datasets)

    fit = PolymlpFitStandardLOOCV(params, data, verbose=True)
    fit.fit()
    train_xy = fit.train_xy
    mlp = fit.best_model

    _, obj = compute_errors(
        mlp=mlp,
        train=data,
        train_xy=train_xy,
        use_cv=True,
        log_energy=False,
        verbose=True,
    )
    tag = "LOOCV:Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(1.7805415513515136e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.0008358087025103993)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(1.3905925966991775e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.0006663275130366884)

    _, obj = compute_errors(
        mlp=mlp,
        train=data,
        train_xy=train_xy,
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

    fit = PolymlpFitStandardUseXLOOCV(params, data, verbose=True)
    fit.fit()
    train_xy = fit.train_xy
    mlp = fit.best_model

    _, obj = compute_errors(
        mlp, data, train_xy=train_xy, use_cv=True, log_energy=False, verbose=True
    )
    tag = "LOOCV:Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(1.77760560018981e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.0008362595489770875)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(1.3890524524384773e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.0006666482589386611)

    _, obj = compute_errors(
        mlp,
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
    assert errors[tag]["energy"] == pytest.approx(1.7723796993157736e-06)
    assert errors[tag]["force"] == pytest.approx(0.0008471005821760739)
    assert errors[tag]["energy_mae"] == pytest.approx(1.381235266206815e-06)
    assert errors[tag]["force_mae"] == pytest.approx(0.0006750500787842289)


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
