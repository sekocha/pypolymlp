"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_sscha():
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Sr", "Ti", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[1.0, 7.0, 7],
        atomic_energy=[0.0, 0.0, 0.0],
        reg_alpha_params=(-1, 3, 10),
    )
    yamlfiles = sorted(glob.glob(str(cwd) + "/data-sscha-SrTiO3/sscha_results_*.yaml"))
    polymlp.set_datasets_sscha(yamlfiles)
    polymlp.run(verbose=True)

    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(0.00014625472319267568, rel=1e-3)
    assert error_test1["force"] == pytest.approx(0.0011131736202838357, rel=1e-3)
    assert error_train1["energy"] == pytest.approx(0.00015277899201758427, rel=1e-3)
    assert error_train1["force"] == pytest.approx(0.0010830176298444144, rel=1e-3)


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
        reg_alpha_params=(-1, 3, 10),
    )
    yamlfiles = sorted(glob.glob(str(cwd) + "/data-electron-Ti/electron-*.yaml"))
    polymlp.set_datasets_electron(yamlfiles, temperature=500, train_ratio=0.8)
    polymlp.run(verbose=True)

    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(1.4650056622179602e-05, rel=1e-2)
    assert error_train1["energy"] == pytest.approx(4.567949042495626e-06, rel=1e-2)
