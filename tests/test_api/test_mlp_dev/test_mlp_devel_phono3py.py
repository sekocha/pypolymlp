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


def test_mlp_dev_phono3py_yaml():
    """Test MLP development using phono3py yaml from input file."""
    infile = str(cwd / "polymlp.in.phono3py.Si")
    pypolymlp = _run_fit(infile)

    tag_train = "Train_Data_from_files"
    tag_test = "Test_Data_from_files"

    assert pypolymlp.n_features == 168
    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]

    assert error_test["energy"] == pytest.approx(6.674941030263998e-06, abs=1e-7)
    assert error_test["force"] == pytest.approx(0.0029171266255415976, abs=1e-6)
    assert error_train["energy"] == pytest.approx(5.847092143276064e-06, abs=1e-7)
    assert error_train["force"] == pytest.approx(0.0028245308500032713, abs=1e-6)
