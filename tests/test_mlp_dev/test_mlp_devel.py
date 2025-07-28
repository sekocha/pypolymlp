"""Tests of polynomial MLP development"""

import glob
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _parse_data(files: str):
    """Parse input files and DFT data."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, prefix=str(cwd))
    pypolymlp.load_datasets(train_ratio=0.9)
    return pypolymlp


def _run_fit(files: str):
    """Run fitting using sequential procedure."""
    pypolymlp = _parse_data(files)
    pypolymlp.fit(batch_size=1000)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _run_fit_standard(files: str):
    """Run fitting using sequential procedure."""
    pypolymlp = _parse_data(files)
    pypolymlp.fit_standard()
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _check_errors_single_dataset_MgO(error_train1, error_test1):

    assert error_test1["energy"] == pytest.approx(5.7907010720826916e-05, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.004816151341623647, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.015092680508112657, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(3.149612103914911e-05, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.0038287840554079465, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.015299229871848218, abs=1e-5)


def _check_errors_multiple_datasets_MgO(
    error_train1,
    error_train2,
    error_test1,
    error_test2,
):
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


def _check_errors_hybrid_single_dataset_MgO(error_train1, error_test1):
    assert error_test1["energy"] == pytest.approx(0.00018507052155432584, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.005399999125414221, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.00413656161693271, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.00022043275013749415, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.002498221077225991, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.00414147022792633, abs=1e-5)


def _check_errors_pair_single_dataset_MgO(error_train1, error_test1):
    assert error_test1["energy"] == pytest.approx(0.0005652094573942398, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.03225765365792877, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.008968003568397519, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.0004734107230618123, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.028966260803851184, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.008789528675929179, abs=1e-5)


def test_mlp_devel_pair_single_dataset():

    file = str(cwd) + "/polymlp.in.pair.single.MgO"
    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

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


def test_mlp_devel_hybrid_single_dataset():

    files = sorted(glob.glob(str(cwd) + "/polymlp*_hybrid.in"))
    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(files)
    assert pypolymlp.n_features == 5736
    _check_errors_hybrid_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_hybrid_single_dataset_standard():

    files = sorted(glob.glob(str(cwd) + "/polymlp*_hybrid.in"))
    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit_standard(files)
    assert pypolymlp.n_features == 5736
    _check_errors_hybrid_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_single_dataset():

    file = str(cwd) + "/polymlp.in.single.MgO"
    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"

    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 1899
    _check_errors_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )

    pypolymlp = _run_fit_standard(file)
    assert pypolymlp.n_features == 1899
    _check_errors_single_dataset_MgO(
        pypolymlp.summary.error_train[tag_train],
        pypolymlp.summary.error_test[tag_test],
    )


def test_mlp_devel_multiple_datasets():

    file = str(cwd) + "/polymlp.in.multi.MgO"
    tag_train1 = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-MgO/vaspruns/test2/vasprun-*.xml.polymlp"

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


def test_mlp_devel_distance():

    file = str(cwd) + "/polymlp.in.SrTiO3.gtinv.distance"
    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 5452

    tag_train = "data-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test = "data-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = pypolymlp.summary.error_train[tag_train]
    error_test1 = pypolymlp.summary.error_test[tag_test]

    assert error_test1["energy"] == pytest.approx(0.0011914132092445697, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.02750490198874777, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.0015997025381622896, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01742941204519919, abs=1e-6)


def test_mlp_devel_distance_pair():

    file = str(cwd) + "/polymlp.in.SrTiO3.pair.distance"
    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 695

    tag_train = "data-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test = "data-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = pypolymlp.summary.error_train[tag_train]
    error_test1 = pypolymlp.summary.error_test[tag_test]

    assert error_test1["energy"] == pytest.approx(0.002675778970795183, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.13474707920071752, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.002882025973254201, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.11969042804382464, abs=1e-6)


def test_mlp_devel_hybrid_flexible_alloy():
    """Test mlp development of hybrid and flexible model in alloy."""
    files = sorted(glob.glob(str(cwd) + "/polymlp.in.Ag-Au.*"))
    pypolymlp = _run_fit(files)
    error_train = pypolymlp.summary.error_train
    error_test = pypolymlp.summary.error_test

    assert pypolymlp.n_features == 790

    tag_train1 = "data-Ag-Au/vaspruns/train-disp1/*.polymlp"
    tag_train2 = "data-Ag-Au/vaspruns/train-standard-Ag1/*.polymlp"
    tag_train3 = "data-Ag-Au/vaspruns/train-standard-Au1/*.polymlp"
    tag_test1 = "data-Ag-Au/vaspruns/test-disp1/*.polymlp"
    tag_test2 = "data-Ag-Au/vaspruns/test-standard-Ag1/*.polymlp"
    tag_test3 = "data-Ag-Au/vaspruns/test-standard-Au1/*.polymlp"
    error_train1 = error_train[tag_train1]
    error_train2 = error_train[tag_train2]
    error_train3 = error_train[tag_train3]
    error_test1 = error_test[tag_test1]
    error_test2 = error_test[tag_test2]
    error_test3 = error_test[tag_test3]

    assert error_test1["energy"] == pytest.approx(0.005856437090626224, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.03669204873660227, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.10038157705917868, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(0.005714896601496177, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.03787574853676284, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.09112941418627805, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(0.016152217081171592, abs=1e-8)
    assert error_test2["force"] == pytest.approx(0.06657513354721871, abs=1e-6)
    assert error_test3["energy"] == pytest.approx(0.03960687938768066, abs=1e-8)
    assert error_test3["force"] == pytest.approx(0.040258801388977375, abs=1e-6)

    assert error_train2["energy"] == pytest.approx(0.012298087188725068, abs=1e-8)
    assert error_train2["force"] == pytest.approx(0.05182914502932192, abs=1e-6)
    assert error_train3["energy"] == pytest.approx(0.004038061027003977, abs=1e-8)
    assert error_train3["force"] == pytest.approx(0.03427719245990994, abs=1e-6)


def test_mlp_devel_hybrid_flexible():

    files = sorted(
        glob.glob(str(cwd) + "/infile-hybrid-flexible/polymlp*_hybrid_flexible.in")
    )
    pypolymlp = _run_fit(files)
    assert pypolymlp.n_features == 7672
    error_train = pypolymlp.summary.error_train
    error_test = pypolymlp.summary.error_test

    tag_train1 = "data-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test1 = "data-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = error_train[tag_train1]
    error_test1 = error_test[tag_test1]

    assert error_train1["energy"] == pytest.approx(0.0015957929458760023, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01733181715196406, abs=1e-6)

    assert error_test1["energy"] == pytest.approx(0.0011686020194212627, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.026877376582754797, abs=1e-6)
