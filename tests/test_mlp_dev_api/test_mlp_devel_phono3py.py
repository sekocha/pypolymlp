"""Tests of polynomial MLP development using yaml."""

from pathlib import Path
from typing import Union

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _run_fit(files: Union[str, list]):
    """Run fitting."""
    pypolymlp = Pypolymlp(verbose=True)
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.fit(batch_size=1000)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _check_errors_phono3py_yaml(error_train: dict, error_test: dict):
    """Check errors for polymlp from phono3py.yaml."""
    assert error_test["energy"] == pytest.approx(1.8102004041582893e-06, rel=1e-2)
    assert error_test["force"] == pytest.approx(0.0008393682108550619, rel=1e-3)
    assert error_train["energy"] == pytest.approx(1.7675397079081478e-06, rel=1e-2)
    assert error_train["force"] == pytest.approx(0.0008322836157656117, rel=1e-3)


def test_mlp_dev_phono3py_yaml():
    """Test MLP development using phono3py yaml from input file."""
    infile = str(cwd / "polymlp.in.phono3py.Si")
    pypolymlp = _run_fit(infile)

    tag_train = "Train_Data_from_files"
    tag_test = "Test_Data_from_files"

    assert pypolymlp.n_features == 168
    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]

    _check_errors_phono3py_yaml(error_train, error_test)
