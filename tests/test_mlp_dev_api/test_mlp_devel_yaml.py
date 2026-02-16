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
    tag_train1 = "Train_data-sscha-SrTiO3/sscha_results*.yaml_no_imag"
    tag_test1 = "Test_data-sscha-SrTiO3/sscha_results*.yaml_no_imag"
    tag_train2 = "Train_data-sscha-SrTiO3/sscha_results*.yaml_imag"
    tag_test2 = "Test_data-sscha-SrTiO3/sscha_results*.yaml_imag"

    error_train1 = pypolymlp.summary.error_train[tag_train1]
    error_test1 = pypolymlp.summary.error_test[tag_test1]
    error_train2 = pypolymlp.summary.error_train[tag_train2]
    error_test2 = pypolymlp.summary.error_test[tag_test2]

    assert error_test1["energy"] == pytest.approx(0.005428856308307956, rel=1e-3)
    assert error_test1["force"] == pytest.approx(0.0012688163505438013, rel=1e-3)
    assert error_train1["energy"] == pytest.approx(0.01448042213655658, rel=1e-3)
    assert error_train1["force"] == pytest.approx(0.001592800914683042, rel=1e-3)
    assert error_test2["energy"] == pytest.approx(0.01129707208231125, rel=1e-3)
    assert error_test2["force"] == pytest.approx(0.007713906991233589, rel=1e-3)
    assert error_train2["energy"] == pytest.approx(0.006487735283537794, rel=1e-3)
    assert error_train2["force"] == pytest.approx(0.005020346796099093, rel=1e-3)


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
