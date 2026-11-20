"""Tests of polynomial MLP development"""

from pathlib import Path

import pytest
from utils import _run_fit, _run_fit_standard

cwd = Path(__file__).parent


def _check_errors_MgO(error_train1: dict, error_test1: dict):
    """Assert errors of MLP from single dataset in MgO."""
    assert error_test1["energy"] == pytest.approx(0.0006141869971372666, abs=1e-7)
    assert error_test1["force"] == pytest.approx(0.04027064539763247, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.003613562567929602, abs=1e-5)
    assert error_train1["energy"] == pytest.approx(0.0005624357894282462, abs=1e-7)
    assert error_train1["force"] == pytest.approx(0.03777769962092713, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.0029785865667518734, abs=1e-5)


def _check_errors_multi_MgO(
    error_train1: dict,
    error_train2: dict,
    error_test1: dict,
    error_test2: dict,
):
    """Assert errors of MLP from multiple datasets in MgO."""

    assert error_test1["energy"] == pytest.approx(0.002835635948786566, abs=1e-7)
    assert error_test1["force"] == pytest.approx(0.1459950979765284, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.012318016498120329, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(0.029757201553560434, abs=1e-7)
    assert error_test2["force"] == pytest.approx(1.2226684184767327, abs=1e-6)
    assert error_test2["stress"] == pytest.approx(0.10847019392485471, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.0025963337173766426, abs=1e-7)
    assert error_train1["force"] == pytest.approx(0.1379593663431816, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.013496672210720002, abs=1e-5)

    assert error_train2["energy"] == pytest.approx(0.028712075597278522, abs=1e-7)
    assert error_train2["force"] == pytest.approx(1.081373600822157, abs=1e-6)
    assert error_train2["stress"] == pytest.approx(0.0877823209080126, abs=1e-5)


def test_mlp_devel_pair_weight():
    """Test pair features."""

    file = str(cwd) + "/polymlp.in.vasp.pair.single.weight.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 324
    _check_errors_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 324
    _check_errors_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_pair_multi_weight():
    """Test multiple datasets."""

    file = str(cwd) + "/polymlp.in.vasp.multi.MgO"
    file = str(cwd) + "/polymlp.in.vasp.pair.multi.weight.MgO"
    tag_train1 = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-vasp-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-vasp-MgO/vaspruns/test2/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 324
    _check_errors_multi_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 324
    _check_errors_multi_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )


def test_mlp_devel_pair_grad():
    """Test pair features."""

    file = str(cwd) + "/polymlp.in.vasp.pair.single.grad.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 324
    _check_errors_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 324
    _check_errors_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_pair_multi_grad():
    """Test pair features."""

    file = str(cwd) + "/polymlp.in.vasp.pair.multi.grad.MgO"
    tag_train1 = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-vasp-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-vasp-MgO/vaspruns/test2/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 324
    _check_errors_multi_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 324
    _check_errors_multi_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )


def test_mlp_devel_pair_cv():
    """Test pair features."""

    file = str(cwd) + "/polymlp.in.vasp.pair.single.cv.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "LOOCV:data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 260
    error_train = pypolymlp.summary.error_train[tag_train]
    error_test = pypolymlp.summary.error_test[tag_test]

    assert error_train["energy"] == pytest.approx(0.0004973784874550861, abs=1e-7)
    assert error_train["force"] == pytest.approx(0.031439874782688546, abs=1e-6)
    assert error_train["stress"] == pytest.approx(0.00715314309722723, abs=1e-5)
    assert error_test["energy"] == pytest.approx(0.0005111390415331175, abs=1e-7)
    assert error_test["force"] == pytest.approx(0.03171257144135115, abs=1e-6)
    assert error_test["stress"] == pytest.approx(0.007190104351933416, abs=1e-5)
