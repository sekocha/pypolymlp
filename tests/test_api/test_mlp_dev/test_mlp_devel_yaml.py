"""Tests of polynomial MLP development using yaml."""

from pathlib import Path
from typing import Union

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _run_fit(files: Union[str, list]):
    """Run fitting."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.fit(batch_size=1000)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def test_mlp_dev_sscha_yaml():
    """Test MLP development for sscha from input file."""
    infile = str(cwd / "polymlp.in.sscha.SrTiO3")
    pypolymlp = _run_fit(infile)

    assert pypolymlp.n_features == 1998
    tag_train = "Train_data-sscha-SrTiO3/sscha_results*.yaml"
    tag_test = "Test_data-sscha-SrTiO3/sscha_results*.yaml"

    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]

    assert error_test["energy"] == pytest.approx(0.005427826614561482, abs=1e-7)
    assert error_test["force"] == pytest.approx(0.0012648715357185438, abs=1e-6)
    assert error_train["energy"] == pytest.approx(0.014479654419790216, abs=1e-7)
    assert error_train["force"] == pytest.approx(0.0015916710361414581, abs=1e-6)


def test_mlp_dev_electron_yaml():
    """Test MLP development for electronic free energy from input file."""
    infile = str(cwd / "polymlp.in.electron.Ti")
    pypolymlp = _run_fit(infile)

    assert pypolymlp.n_features == 75
    tag_train = "Train_data-electron-Ti/electron-*.yaml"
    tag_test = "Test_data-electron-Ti/electron-*.yaml"

    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]

    assert error_test["energy"] == pytest.approx(4.185723757235823e-05, abs=1e-7)
    assert error_train["energy"] == pytest.approx(9.392970069536276e-06, abs=1e-7)
