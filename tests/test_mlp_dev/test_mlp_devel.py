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


def _check_errors_single_dataset_MgO(error_train1, error_test1):
    assert error_train1["energy"] == pytest.approx(3.1791594630511444e-05, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.003822251017162934, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.015284354036260491, abs=1e-5)

    assert error_test1["energy"] == pytest.approx(6.0128773079683234e-05, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.004820856779955612, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.015064657699737062, abs=1e-5)


def _check_errors_multiple_datasets_MgO(
    error_train1,
    error_train2,
    error_test1,
    error_test2,
):
    assert error_train1["energy"] == pytest.approx(2.7039999690544686e-4, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.015516294909701862, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.01024817, abs=1e-5)

    assert error_train2["energy"] == pytest.approx(6.587197553779358e-3, abs=1e-8)
    assert error_train2["force"] == pytest.approx(0.3157543533276883, abs=1e-6)
    assert error_train2["stress"] == pytest.approx(0.03763428, abs=1e-5)

    assert error_test1["energy"] == pytest.approx(2.2913988829580454e-4, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.01565802222609203, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.01109818, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(6.022423646649027e-3, abs=1e-8)
    assert error_test2["force"] == pytest.approx(0.40361027823620954, abs=1e-6)
    assert error_test2["stress"] == pytest.approx(0.03756416, abs=1e-5)


def _check_errors_hybrid_single_dataset_MgO(error_train1, error_test1):
    assert error_train1["energy"] == pytest.approx(0.00022116079973644265, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.002724472551198224, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.0042136414619126225, abs=1e-5)

    assert error_test1["energy"] == pytest.approx(0.00018904268491116763, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.006041215496572942, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.004267059946631895, abs=1e-5)


def _check_errors_pair_single_dataset_MgO(error_train1, error_test1):
    assert error_train1["energy"] == pytest.approx(0.0004734107230618123, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.028966260803851184, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.008789528675929179, abs=1e-5)

    assert error_test1["energy"] == pytest.approx(0.0005652094573942398, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.03225765365792877, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.008968003568397519, abs=1e-5)


def _parse_data(filename):
    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(filename, verbose=True, prefix=str(cwd))
    polymlp_in.parse_datasets()
    return polymlp_in


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
