"""Tests of polynomial MLP development"""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _run_fit(files: str):
    """Run fitting using sequential procedure."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.fit(batch_size=1000)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _check_errors_spin_Fe(error_train: dict, error_test: dict):
    """Assert errors of MLP from spin-polarized dataset in Fe."""
    assert error_test["energy"] == pytest.approx(0.00513671462938704, rel=1e-3)
    assert error_test["force"] == pytest.approx(0.17017223118191127, rel=1e-3)
    assert error_test["stress"] == pytest.approx(0.06425086194654982, rel=1e-3)
    assert error_test["energy_mae"] == pytest.approx(0.002852092461563456, rel=1e-3)
    assert error_test["force_mae"] == pytest.approx(0.09346174770492068, rel=1e-3)
    assert error_test["stress_mae"] == pytest.approx(0.03412023832035358, rel=1e-3)

    assert error_train["energy"] == pytest.approx(0.004681448992076226, rel=1e-3)
    assert error_train["force"] == pytest.approx(0.15444495672711753, rel=1e-3)
    assert error_train["stress"] == pytest.approx(0.06938864207245748, rel=1e-3)
    assert error_train["energy_mae"] == pytest.approx(0.0026610109108202344, rel=1e-3)
    assert error_train["force_mae"] == pytest.approx(0.08755942539823292, rel=1e-3)
    assert error_train["stress_mae"] == pytest.approx(0.035308173211984914, rel=1e-3)


def _check_errors_hybrid_spin_Fe(error_train: dict, error_test: dict):
    """Assert errors of hybrid MLP from spin-polarized dataset in Fe."""
    assert error_test["energy"] == pytest.approx(0.005057471891220216, rel=1e-3)
    assert error_test["force"] == pytest.approx(0.1699281904979255, rel=1e-3)
    assert error_test["stress"] == pytest.approx(0.06492116053052065, rel=1e-3)
    assert error_test["energy_mae"] == pytest.approx(0.002759033864812692, rel=1e-3)
    assert error_test["force_mae"] == pytest.approx(0.09299697918476171, rel=1e-3)
    assert error_test["stress_mae"] == pytest.approx(0.03470003471157489, rel=1e-3)

    assert error_train["energy"] == pytest.approx(0.004652761212721123, rel=1e-3)
    assert error_train["force"] == pytest.approx(0.153731572811346, rel=1e-3)
    assert error_train["stress"] == pytest.approx(0.06942843178937845, rel=1e-3)
    assert error_train["energy_mae"] == pytest.approx(0.002656962833666678, rel=1e-3)
    assert error_train["force_mae"] == pytest.approx(0.08701361362317887, rel=1e-3)
    assert error_train["stress_mae"] == pytest.approx(0.03523283227722057, rel=1e-3)


def test_mlp_devel_spin_Fe():
    """Test mlp development in spin-polarized FE."""

    file = str(cwd) + "/polymlp.in.vasp.spin.Fe.1"
    tag_train = "data-vasp-spin-Fe/vaspruns/train/*.xml"
    tag_test = "data-vasp-spin-Fe/vaspruns/test/*.xml"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 2439

    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]
    _check_errors_spin_Fe(error_train, error_test)


def test_mlp_devel_hybrid_spin_Fe():
    """Test mlp development (hybrid) in spin-polarized FE."""

    file1 = str(cwd) + "/polymlp.in.vasp.spin.Fe.1"
    file2 = str(cwd) + "/polymlp.in.vasp.spin.Fe.2"
    files = [file1, file2]
    tag_train = "data-vasp-spin-Fe/vaspruns/train/*.xml"
    tag_test = "data-vasp-spin-Fe/vaspruns/test/*.xml"

    pypolymlp = _run_fit(files)
    assert pypolymlp.n_features == 2549

    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]
    print(error_test)
    print(error_train)
    _check_errors_hybrid_spin_Fe(error_train, error_test)
