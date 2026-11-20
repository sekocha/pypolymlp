"""Tests of polynomial MLP development using yaml."""

from pathlib import Path

import pytest
from utils import _run_fit

cwd = Path(__file__).parent


def _check_errors_phono3py_yaml(error_train: dict, error_test: dict):
    """Check errors for polymlp from phono3py.yaml."""
    assert error_test["energy"] == pytest.approx(2.0665116376430617e-06, rel=1e-2)
    assert error_test["force"] == pytest.approx(0.0009180170468419943, rel=1e-2)
    assert error_train["energy"] == pytest.approx(1.925059100477335e-06, rel=1e-2)
    assert error_train["force"] == pytest.approx(0.0009113011502068905, rel=1e-2)


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
