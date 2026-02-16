"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_sscha():
    """Test API to develop MLP for sscha free energy."""
    polymlp = Pypolymlp(verbose=True)
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
        include_stress=False,
    )
    yamlfiles = sorted(glob.glob(str(cwd) + "/data-sscha-SrTiO3/sscha_results_*.yaml"))
    polymlp.set_datasets_sscha(yamlfiles)
    polymlp.run()

    error_train1 = polymlp.summary.error_train["data1_no_imag"]
    error_test1 = polymlp.summary.error_test["data2_no_imag"]

    error_train2 = polymlp.summary.error_train["data1_imag"]
    error_test2 = polymlp.summary.error_test["data2_imag"]

    assert error_test1["energy"] == pytest.approx(0.0001437891687043342, rel=1e-3)
    assert error_test1["force"] == pytest.approx(0.001112602030598342, rel=1e-3)
    assert error_train1["energy"] == pytest.approx(0.0001533446958244101, rel=1e-3)
    assert error_train1["force"] == pytest.approx(0.0010834715929883944, rel=1e-3)

    assert error_test2["energy"] == pytest.approx(0.0030792057826591964, rel=1e-3)
    assert error_test2["force"] == pytest.approx(0.0056073282503636915, rel=1e-3)
    assert error_train2["energy"] == pytest.approx(0.0033314139457414366, rel=1e-3)
    assert error_train2["force"] == pytest.approx(0.004875165188815314, rel=1e-3)


def test_mlp_devel_api_electron():
    """Test API to develop MLP for electronic free energy."""
    polymlp = Pypolymlp(verbose=True)
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
    polymlp.run()

    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(1.4650056622179602e-05, rel=1e-2)
    assert error_train1["energy"] == pytest.approx(4.567949042495626e-06, rel=1e-2)
