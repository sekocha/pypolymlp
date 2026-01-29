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


def test_mlp_devel_distance():

    file = str(cwd) + "/polymlp.in.vasp.gtinv.distance.SrTiO3"
    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 5452

    tag_train = "data-vasp-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test = "data-vasp-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = pypolymlp.summary.error_train[tag_train]
    error_test1 = pypolymlp.summary.error_test[tag_test]

    assert error_test1["energy"] == pytest.approx(0.0011914132092445697, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.02750490198874777, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.0015997025381622896, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01742941204519919, abs=1e-6)


def test_mlp_devel_distance_pair():

    file = str(cwd) + "/polymlp.in.vasp.pair.distance.SrTiO3"
    pypolymlp = _run_fit(file)
    assert pypolymlp.n_features == 695

    tag_train = "data-vasp-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test = "data-vasp-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = pypolymlp.summary.error_train[tag_train]
    error_test1 = pypolymlp.summary.error_test[tag_test]

    assert error_test1["energy"] == pytest.approx(0.002675778970795183, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.13474707920071752, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.002882025973254201, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.11969042804382464, abs=1e-6)


def test_mlp_devel_hybrid_flexible():
    """Test mlp development of hybrid and flexible model in SrTiO3."""
    files = sorted(
        glob.glob(
            str(cwd) + "/infile-hybrid-flexible-SrTiO3/polymlp*_hybrid_flexible.in"
        )
    )
    pypolymlp = _run_fit(files)
    assert pypolymlp.n_features == 7672
    error_train = pypolymlp.summary.error_train
    error_test = pypolymlp.summary.error_test

    tag_train1 = "data-vasp-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test1 = "data-vasp-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = error_train[tag_train1]
    error_test1 = error_test[tag_test1]

    assert error_train1["energy"] == pytest.approx(0.0015957929458760023, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01733181715196406, abs=1e-6)

    assert error_test1["energy"] == pytest.approx(0.0011686020194212627, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.026877376582754797, abs=1e-6)


def test_mlp_devel_hybrid_flexible_alloy():
    """Test mlp development of hybrid and flexible model in alloy."""
    files = sorted(glob.glob(str(cwd) + "/polymlp.in.vasp.Ag-Au.*"))
    pypolymlp = _run_fit(files)
    error_train = pypolymlp.summary.error_train
    error_test = pypolymlp.summary.error_test

    assert pypolymlp.n_features == 790

    tag_train1 = "data-vasp-Ag-Au/vaspruns/train-disp1/*.polymlp"
    tag_train2 = "data-vasp-Ag-Au/vaspruns/train-standard-Ag1/*.polymlp"
    tag_train3 = "data-vasp-Ag-Au/vaspruns/train-standard-Au1/*.polymlp"
    tag_test1 = "data-vasp-Ag-Au/vaspruns/test-disp1/*.polymlp"
    tag_test2 = "data-vasp-Ag-Au/vaspruns/test-standard-Ag1/*.polymlp"
    tag_test3 = "data-vasp-Ag-Au/vaspruns/test-standard-Au1/*.polymlp"
    error_train1 = error_train[tag_train1]
    error_train2 = error_train[tag_train2]
    error_train3 = error_train[tag_train3]
    error_test1 = error_test[tag_test1]
    error_test2 = error_test[tag_test2]
    error_test3 = error_test[tag_test3]

    etol, ftol, stol = 1e-2, 5e-3, 5e-3
    assert error_test1["energy"] == pytest.approx(0.005856437090626224, rel=etol)
    assert error_test1["force"] == pytest.approx(0.03669204873660227, rel=ftol)
    assert error_test1["stress"] == pytest.approx(0.10038157705917868, rel=stol)

    assert error_train1["energy"] == pytest.approx(0.005714896601496177, rel=etol)
    assert error_train1["force"] == pytest.approx(0.03787574853676284, rel=ftol)
    assert error_train1["stress"] == pytest.approx(0.09112941418627805, rel=stol)

    assert error_test2["energy"] == pytest.approx(0.016152217081171592, rel=etol)
    assert error_test2["force"] == pytest.approx(0.06657513354721871, rel=ftol)
    assert error_test3["energy"] == pytest.approx(0.03960687938768066, rel=etol)
    assert error_test3["force"] == pytest.approx(0.040258801388977375, rel=ftol)

    assert error_train2["energy"] == pytest.approx(0.012298087188725068, rel=etol)
    assert error_train2["force"] == pytest.approx(0.05182914502932192, rel=ftol)
    assert error_train3["energy"] == pytest.approx(0.004038061027003977, rel=etol)
    assert error_train3["force"] == pytest.approx(0.03427719245990994, rel=ftol)
