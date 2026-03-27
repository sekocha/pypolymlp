"""Tests of polynomial MLP development using yaml."""

from pathlib import Path
from typing import Union

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _run_fit(files: Union[str, list]):
    """Run fitting."""
    pypolymlp = Pypolymlp(verbose=True)
    pypolymlp.load_parameter_file(
        files, train_ratio=0.9, prefix=str(cwd / "data-openmx-AgC")
    )
    pypolymlp.fit(batch_size=1000)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _assert_AgC(error_train: dict, error_test: dict):
    """Assert regression results."""
    assert error_test["energy"] == pytest.approx(0.00018557502400592924, rel=1e-2)
    assert error_test["force"] == pytest.approx(0.01363102581104331, rel=1e-3)
    assert error_train["energy"] == pytest.approx(0.0001801334904126147, rel=1e-2)
    assert error_train["force"] == pytest.approx(0.012532615917227246, rel=1e-3)


def _assert_AgC_hybrid(error_train: dict, error_test: dict):
    """Assert regression results."""
    assert error_test["energy"] == pytest.approx(0.00016563370642374878, rel=1e-2)
    assert error_test["force"] == pytest.approx(0.011224567747625657, rel=1e-3)
    assert error_train["energy"] == pytest.approx(0.000163876427103251, rel=1e-2)
    assert error_train["force"] == pytest.approx(0.010396168543175642, rel=1e-3)


def test_mlp_dev_openmx():
    """Test MLP development using openmx data."""
    infile = str(cwd / "data-openmx-AgC/polymlp.in")
    pypolymlp = _run_fit(infile)

    tag_train = "Train_sample.md"
    tag_test = "Test_sample.md"

    assert pypolymlp.n_features == 1660
    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]
    _assert_AgC(error_train, error_test)


def test_mlp_dev_openmx_hybrid():
    """Test MLP development (hybrid) using openmx data."""
    infile = [
        str(cwd / "data-openmx-AgC/polymlp.in"),
        str(cwd / "data-openmx-AgC/polymlp.in.2"),
    ]
    pypolymlp = _run_fit(infile)

    tag_train = "Train_sample.md"
    tag_test = "Test_sample.md"

    assert pypolymlp.n_features == 4099
    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]
    _assert_AgC_hybrid(error_train, error_test)
