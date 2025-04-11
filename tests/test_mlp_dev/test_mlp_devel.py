"""Tests of polynomial MLP development"""

import glob
from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression

cwd = Path(__file__).parent


def _parse_data(filename):
    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(filename, prefix=str(cwd))
    polymlp_in.parse_datasets()
    return polymlp_in


def test_mlp_devel_hybrid_flexible_alloy():

    files = sorted(glob.glob(str(cwd) + "/polymlp.in.Ag-Au.*"))
    polymlp_in = _parse_data(files)
    assert polymlp_in.n_features == 790

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train1 = "data-Ag-Au/vaspruns/train-disp1/*.polymlp"
    tag_train2 = "data-Ag-Au/vaspruns/train-standard-Ag1/*.polymlp"
    tag_train3 = "data-Ag-Au/vaspruns/train-standard-Au1/*.polymlp"
    tag_test1 = "data-Ag-Au/vaspruns/test-disp1/*.polymlp"
    tag_test2 = "data-Ag-Au/vaspruns/test-standard-Ag1/*.polymlp"
    tag_test3 = "data-Ag-Au/vaspruns/test-standard-Au1/*.polymlp"
    error_train1 = acc.error_train_dict[tag_train1]
    error_train2 = acc.error_train_dict[tag_train2]
    error_train3 = acc.error_train_dict[tag_train3]
    error_test1 = acc.error_test_dict[tag_test1]
    error_test2 = acc.error_test_dict[tag_test2]
    error_test3 = acc.error_test_dict[tag_test3]

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
    polymlp_in = _parse_data(files)
    assert polymlp_in.n_features == 7672

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train1 = "data-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test1 = "data-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = acc.error_train_dict[tag_train1]
    error_test1 = acc.error_test_dict[tag_test1]

    assert error_train1["energy"] == pytest.approx(0.0015957929458760023, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01733181715196406, abs=1e-6)

    assert error_test1["energy"] == pytest.approx(0.0011686020194212627, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.026877376582754797, abs=1e-6)


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


def test_mlp_devel_pair_single_dataset_seq():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.pair.single.MgO")
    assert polymlp_in.n_features == 324

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train]
    error_test1 = acc.error_test_dict[tag_test]
    _check_errors_pair_single_dataset_MgO(error_train1, error_test1)


def test_mlp_devel_hybrid_single_dataset_seq():

    files = sorted(glob.glob(str(cwd) + "/polymlp*_hybrid.in"))
    polymlp_in = _parse_data(files)
    assert polymlp_in.n_features == 5736

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train1 = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test1 = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train1]
    error_test1 = acc.error_test_dict[tag_test1]
    _check_errors_hybrid_single_dataset_MgO(
        error_train1,
        error_test1,
    )


def test_mlp_devel_single_dataset_seq():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.single.MgO")
    assert polymlp_in.n_features == 1899

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train]
    error_test1 = acc.error_test_dict[tag_test]
    _check_errors_single_dataset_MgO(error_train1, error_test1)


def test_mlp_devel_single_dataset_noseq():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.single.MgO")
    assert polymlp_in.n_features == 1899

    polymlp = PolymlpDevDataXY(polymlp_in).run()
    reg = Regression(polymlp).fit(seq=False)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_test = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train]
    error_test1 = acc.error_test_dict[tag_test]
    _check_errors_single_dataset_MgO(error_train1, error_test1)


def test_mlp_devel_multiple_datasets_seq():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.multi.MgO")
    assert polymlp_in.n_features == 1899

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train1 = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-MgO/vaspruns/test2/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train1]
    error_train2 = acc.error_train_dict[tag_train2]
    error_test1 = acc.error_test_dict[tag_test1]
    error_test2 = acc.error_test_dict[tag_test2]
    _check_errors_multiple_datasets_MgO(
        error_train1, error_train2, error_test1, error_test2
    )


def test_mlp_devel_multiple_datasets_noseq():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.multi.MgO")
    assert polymlp_in.n_features == 1899

    polymlp = PolymlpDevDataXY(polymlp_in).run()
    reg = Regression(polymlp).fit(seq=False)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train1 = "data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    tag_train2 = "data-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    tag_test1 = "data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    tag_test2 = "data-MgO/vaspruns/test2/vasprun-*.xml.polymlp"
    error_train1 = acc.error_train_dict[tag_train1]
    error_train2 = acc.error_train_dict[tag_train2]
    error_test1 = acc.error_test_dict[tag_test1]
    error_test2 = acc.error_test_dict[tag_test2]
    _check_errors_multiple_datasets_MgO(
        error_train1, error_train2, error_test1, error_test2
    )


def test_mlp_devel_distance():

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.SrTiO3.gtinv.distance")
    assert polymlp_in.n_features == 5452

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    tag_train = "data-SrTiO3/vaspruns/train1/vasprun.xml.*"
    tag_test = "data-SrTiO3/vaspruns/test1/vasprun.xml.*"
    error_train1 = acc.error_train_dict[tag_train]
    error_test1 = acc.error_test_dict[tag_test]

    assert error_test1["energy"] == pytest.approx(0.0011914132092445697, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.02750490198874777, abs=1e-6)

    assert error_train1["energy"] == pytest.approx(0.0015997025381622896, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01742941204519919, abs=1e-6)

    polymlp_in = _parse_data(str(cwd) + "/polymlp.in.SrTiO3.pair.distance")
    assert polymlp_in.n_features == 695

    polymlp = PolymlpDevDataXYSequential(polymlp_in).run_train(batch_size=1000)
    reg = Regression(polymlp).fit(seq=True, clear_data=True, batch_size=1000)
    acc = PolymlpDevAccuracy(reg)
    acc.compute_error(log_energy=False)

    error_train1 = acc.error_train_dict[tag_train]
    error_test1 = acc.error_test_dict[tag_test]

    assert error_test1["energy"] == pytest.approx(0.002675778970795183, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.13474707920071752, abs=1e-6)

    assert error_train1["energy"] == pytest.approx(0.002882025973254201, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.11969042804382464, abs=1e-6)
