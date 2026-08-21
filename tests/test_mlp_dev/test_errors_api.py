"""Tests of PolymlpEvalAccuracy."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.errors.api_errors import compute_errors, eval_rmse

cwd = Path(__file__).parent


def test_compute_errors(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    obj, _ = compute_errors(mlp_mp_149, datasets, test=datasets, use_cv=False)
    tag = "Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(5.84708583252534e-06)
    assert obj.errors[tag]["force"] == pytest.approx(0.0028245295894005276)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(4.668251936400402e-06)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.002250798354169759)


def test_compute_cv(regdata_mp_149, mlp_mp_149, dataxy_xtx_xty_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    xtx = dataxy_xtx_xty_mp_149.xtx
    inv_xtx = np.linalg.inv(xtx)

    _, obj = compute_errors(mlp_mp_149, datasets, inv_xtx=inv_xtx, use_cv=True)
    tag = "LOOCV:Train_Data_from_files"
    assert obj.errors[tag]["energy"] == pytest.approx(1.8956990735382943e-12)
    assert obj.errors[tag]["force"] == pytest.approx(0.05383648494520971)
    assert obj.errors[tag]["energy_mae"] == pytest.approx(1.5134410137908146e-12)
    assert obj.errors[tag]["force_mae"] == pytest.approx(0.003709359247952824)


def test_eval_rmse(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    errors = eval_rmse(mlp_mp_149, datasets)
    tag = "Train_Data_from_files"
    assert errors[tag]["energy"] == pytest.approx(5.84708583252534e-06)
    assert errors[tag]["force"] == pytest.approx(0.0028245295894005276)
    assert errors[tag]["energy_mae"] == pytest.approx(4.668251936400402e-06)
    assert errors[tag]["force_mae"] == pytest.approx(0.002250798354169759)


def test_class_rmse(regdata_mp_149, mlp_mp_149):
    """Test for compute_errors."""
    _, datasets = regdata_mp_149
    obj, _ = compute_errors(mlp_mp_149, datasets, test=datasets, use_cv=False)
    for k, v in obj.errors.items():
        obj.print_error(v)
    obj.write_error_yaml(filename="tmp.yaml")
    os.remove("tmp.yaml")
