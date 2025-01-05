"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_sscha():
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Zn", "S"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 7],
        atomic_energy=[0.0, 0.0],
        reg_alpha_params=(-3, 3, 10),
    )
    yamlfiles = sorted(glob.glob(str(cwd) + "/data-sscha-ZnS/sscha_results_*.yaml"))
    polymlp.set_datasets_sscha(yamlfiles)
    polymlp.run(verbose=True)

    error_train1 = polymlp.summary.error_train["train_single"]
    error_test1 = polymlp.summary.error_test["test_single"]

    assert error_train1["energy"] == pytest.approx(5.765120561897502e-05, rel=1e-4)
    assert error_test1["energy"] == pytest.approx(7.061448981395381e-05, rel=1e-4)


def test_mlp_devel_api_electron():
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ti"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 6.0, 7],
        atomic_energy=[0],
        reg_alpha_params=(-2, 2, 5),
    )
    yamlfiles = sorted(glob.glob(str(cwd) + "/data-electron-Ti/electron-*.yaml"))
    polymlp.set_datasets_electron(yamlfiles, temperature=500)
    polymlp.run(verbose=True)

    error_train1 = polymlp.summary.error_train["train_single"]
    error_test1 = polymlp.summary.error_test["test_single"]

    assert error_train1["energy"] == pytest.approx(3.496701607548639e-05, rel=1e-4)
    assert error_test1["energy"] == pytest.approx(5.5263412195305506e-06, rel=1e-4)
