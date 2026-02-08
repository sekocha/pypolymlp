"""Tests of polynomial MLP development"""

import glob
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


def _run_fit_standard(files: str):
    """Run fitting using standard procedure."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.fit_standard()
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _check_errors_single_dataset_MgO(
    error_train1: dict, error_test1: dict, use_stress: bool = False
):
    """Assert errors of MLP from single dataset in MgO."""
    assert error_test1["energy"] == pytest.approx(5.7907010720826916e-05, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.004816151341623647, abs=1e-6)
    if use_stress:
        assert error_test1["stress"] == pytest.approx(0.015092680508112657, abs=1e-5)
    assert error_train1["energy"] == pytest.approx(3.149612103914911e-05, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.0038287840554079465, abs=1e-6)
    if use_stress:
        assert error_train1["stress"] == pytest.approx(0.015299229871848218, abs=1e-5)


def _check_errors_multiple_datasets_MgO(
    error_train1: dict,
    error_train2: dict,
    error_test1: dict,
    error_test2: dict,
):
    """Assert errors of MLP from multiple datasets in MgO."""
    assert error_test1["energy"] == pytest.approx(2.2913988829580454e-4, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.01565802222609203, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.01109818, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(6.022423646649027e-3, abs=1e-8)
    assert error_test2["force"] == pytest.approx(0.40361027823620954, abs=1e-6)
    assert error_test2["stress"] == pytest.approx(0.03756416, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(2.7039999690544686e-4, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.015516294909701862, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.01024817, abs=1e-5)

    assert error_train2["energy"] == pytest.approx(6.587197553779358e-3, abs=1e-8)
    assert error_train2["force"] == pytest.approx(0.3157543533276883, abs=1e-6)
    assert error_train2["stress"] == pytest.approx(0.03763428, abs=1e-5)


def _check_errors_single_dataset_MgO_auto(error_train1: dict, error_test1: dict):
    """Assert errors of MLP from single dataset using automatic division in MgO."""
    assert error_test1["energy"] == pytest.approx(3.682827414945049e-05, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.004239957033147398, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(3.288929693819795e-05, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.0038657674044820195, abs=1e-6)


def _check_errors_hybrid_single_dataset_MgO(error_train1: dict, error_test1: dict):
    """Assert errors of hybrid MLP from single dataset in MgO."""
    assert error_test1["energy"] == pytest.approx(0.00018507052155432584, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.005399999125414221, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.00413656161693271, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.00022043275013749415, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.002498221077225991, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.00414147022792633, abs=1e-5)


def _check_errors_pair_single_dataset_MgO(error_train1: dict, error_test1: dict):
    """Assert errors of pair MLP from single dataset in MgO."""
    assert error_test1["energy"] == pytest.approx(0.0005652094573942398, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.03225765365792877, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.008968003568397519, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.0004734107230618123, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.028966260803851184, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.008789528675929179, abs=1e-5)


def _check_errors_md_Cu(error_train: dict, error_test: dict):
    """Assert errors of MLP from MD dataset in Cu."""
    assert error_test["energy"] == pytest.approx(0.0005486086714176058, abs=1e-7)
    assert error_test["force"] == pytest.approx(0.04747553260018069, abs=1e-6)
    assert error_train["energy"] == pytest.approx(0.00044458300300040534, abs=1e-7)
    assert error_train["force"] == pytest.approx(0.04692192616585159, abs=1e-6)


def test_mlp_devel_pair():
    """Test pair features."""

    file = str(cwd) + "/polymlp.in.vasp.pair.single.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 324
    _check_errors_pair_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 324
    _check_errors_pair_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_single_dataset():
    """Test single dataset."""

    file = str(cwd) + "/polymlp.in.vasp.single.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 1899
    _check_errors_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_single_dataset_standard():
    """Test single dataset using standard procedure."""

    file = str(cwd) + "/polymlp.in.vasp.single.MgO"
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 1899
    _check_errors_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_multiple_datasets():
    """Test multiple datasets."""

    file = str(cwd) + "/polymlp.in.vasp.multi.MgO"
    tag_train1 = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-vasp-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-vasp-MgO/vaspruns/test2/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 1899
    _check_errors_multiple_datasets_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 1899
    _check_errors_multiple_datasets_MgO(
        pypolymlp.summary.error_train[tag_train1],
        pypolymlp.summary.error_train[tag_train2],
        pypolymlp.summary.error_test[tag_test1],
        pypolymlp.summary.error_test[tag_test2],
    )


def test_mlp_devel_md():
    """Test md datasets."""
    file = str(cwd) + "/polymlp.in.vasp.md.Cu"
    tag_train = "Train_data-vasp-md-Cu/vasprun.xml"
    tag_test = "Test_data-vasp-md-Cu/vasprun.xml"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 225
    _check_errors_md_Cu(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_hybrid_single_dataset():
    """Test hybrid model with single dataset."""

    files = sorted(glob.glob(str(cwd) + "/infile-hybrid-MgO/polymlp*_hybrid.in"))
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(files)
    assert pypolymlp.n_features == 5736
    _check_errors_hybrid_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_hybrid_single_dataset_standard():
    """Test hybrid model with single dataset using standard method."""

    files = sorted(glob.glob(str(cwd) + "/infile-hybrid-MgO/polymlp*_hybrid.in"))
    tag_train = "data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit_standard(files)
    assert pypolymlp.n_features == 5736
    _check_errors_hybrid_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_single_dataset_autodiv():
    """Test single dataset using auto division method."""

    file = str(cwd) + "/polymlp.in.vasp.single.auto.MgO"
    tag_train = "Train_data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp"
    tag_test = "Test_data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 1899
    error_train1 = pypolymlp.summary.error_train[tag_train]
    error_test1 = pypolymlp.summary.error_test[tag_test]

    _check_errors_single_dataset_MgO_auto(error_train1, error_test1)


def test_mlp_devel_multiple_datasets_autodiv():
    """Test multiple datasets using auto division method."""

    file = str(cwd) + "/polymlp.in.vasp.multi.auto.MgO"
    tag_train1 = "Train_data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp"
    tag_train2 = "Train_data-vasp-MgO/vaspruns/*2/vasprun-*.xml.polymlp"
    tag_test1 = "Test_data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp"
    tag_test2 = "Test_data-vasp-MgO/vaspruns/*2/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 1899
    error_train1 = pypolymlp.summary.error_train[tag_train1]
    error_train2 = pypolymlp.summary.error_train[tag_train2]
    error_test1 = pypolymlp.summary.error_test[tag_test1]
    error_test2 = pypolymlp.summary.error_test[tag_test2]

    assert error_test1["energy"] == pytest.approx(0.0002986093686859434, abs=1e-7)
    assert error_test1["force"] == pytest.approx(0.014689850271230067, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.010884472069561626, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.0002806859760938073, abs=1e-7)
    assert error_train1["force"] == pytest.approx(0.015439039118004505, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.010767897079474016, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(0.00746593853779306, abs=1e-7)
    assert error_test2["force"] == pytest.approx(0.4201408013002017, abs=1e-6)
    assert error_test2["stress"] == pytest.approx(0.043439257875251075, abs=1e-5)

    assert error_train2["energy"] == pytest.approx(0.006914586454907308, abs=1e-7)
    assert error_train2["force"] == pytest.approx(0.354911294393713, abs=1e-6)
    assert error_train2["stress"] == pytest.approx(0.03940844639741093, abs=1e-5)
