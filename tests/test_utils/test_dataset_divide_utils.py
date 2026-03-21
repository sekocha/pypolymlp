"""Tests of functions used for dividing dataset automatically."""

import glob
import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.dataset_divide_utils import (
    _extract_properties_from_dataset,
    _extract_properties_from_dataset_alloy,
    _set_threshold_energy,
    _set_threshold_force,
    _set_threshold_volume,
    _split,
    copy_vaspruns,
    split_datasets,
    split_datasets_alloy,
    split_three_datasets,
    split_two_datasets,
)

cwd = Path(__file__).parent


def test_extract_properties_from_dataset(regdata_mp_149):
    """Test _extract_properties_from_dataset."""
    params, datasets = regdata_mp_149
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(datasets[0])
    assert e_all.shape[0] == 180
    assert f_std_all.shape[0] == 180
    assert vol_all.shape[0] == 180


def test_extract_properties_from_dataset_alloy(regdata_mp_149):
    """Test _extract_properties_from_dataset_alloy."""
    params, datasets = regdata_mp_149
    e_all, f_std_all, comp_ids = _extract_properties_from_dataset_alloy(
        datasets[0],
        ["Si", "C"],
    )
    assert e_all.shape[0] == 180
    assert f_std_all.shape[0] == 180
    assert len(comp_ids) == 1
    for ids in comp_ids.values():
        assert len(ids) == 180


def test_split(regdata_mp_149):
    """Test _split."""
    params, datasets = regdata_mp_149
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(datasets[0])
    groups = _split(e_all, f_std_all, n_divide=5)
    np.testing.assert_equal(groups, 4)


def test_split_datasets(regdata_mp_149):
    """Test split_datasets."""
    params, datasets = regdata_mp_149
    ds = split_datasets(datasets[0], n_divide=5)
    assert len(ds) == 5
    assert len(ds[0].train) == 0
    assert len(ds[0].test) == 0
    assert len(ds[4].train) == 162
    assert len(ds[4].test) == 18


def test_split_datasets_alloy(regdata_mp_149):
    """Test split_datasets_alloy."""
    params, datasets = regdata_mp_149
    ds = split_datasets_alloy(datasets[0], ("Si", "C"), n_divide=3)
    assert len(ds) == 6
    assert len(ds[0].train) == 0
    assert len(ds[0].test) == 0
    assert len(ds[5].train) == 162
    assert len(ds[5].test) == 18


def test_copy_vaspruns():
    """Test copy_vaspruns."""
    path = str(cwd) + "/../test_mlp_dev_api/data-vasp-MgO/vaspruns/test1/"
    vaspruns = sorted(glob.glob(path + "vasprun-*.xml.*"))
    copy_vaspruns(vaspruns, "train1", path_output="tmp")
    shutil.rmtree("tmp")


def test_local_functions(regdata_mp_149):
    """Test _extract_properties_from_dataset."""
    params, datasets = regdata_mp_149
    e_all, f_std_all, vol_all = _extract_properties_from_dataset(datasets[0])

    eth = _set_threshold_energy(e_all=e_all, eth=None, e_ratio=0.25)
    assert eth == pytest.approx(-5.737217256749669)
    eth = _set_threshold_energy(e_all=e_all, eth=None, e_ratio=0.5)
    assert eth == pytest.approx(-5.7371101178743364)

    fth = _set_threshold_force(f_std_all=f_std_all, fth=None, f_ratio=1.0)
    assert fth == pytest.approx(0.2801843522626887)
    fth = _set_threshold_force(f_std_all=f_std_all, fth=None, f_ratio=0.5)
    assert fth == pytest.approx(0.14009217613134434)

    volth = _set_threshold_volume(vol_all=vol_all, volth=None, vol_ratio=2.0)
    assert volth == pytest.approx(40.12671069799894)
    volth = _set_threshold_volume(vol_all=vol_all, volth=None, vol_ratio=3.0)
    assert volth == pytest.approx(60.19006604699841)


def test_split_two_datasets(regdata_mp_149):
    """Test split_two_datasets."""
    params, datasets = regdata_mp_149
    train1, train2, test1, test2 = split_two_datasets(
        datasets[0].dft,
        eth=None,
        fth=None,
        volth=None,
        e_ratio=1.0,
        f_ratio=1.0,
        vol_ratio=1.5,
    )
    assert len(train1) == 14
    assert len(train2) == 148
    assert len(test1) == 1
    assert len(test2) == 17


def test_split_three_datasets(regdata_mp_149):
    """Test split_three_datasets."""
    params, datasets = regdata_mp_149
    train1, train2, train0, test1, test2, test0 = split_three_datasets(
        datasets[0].dft,
        eth=None,
        fth=None,
        volth=None,
        e_ratio=1.0,
        f_ratio=1.0,
        vol_ratio=1.5,
    )
    assert len(train1) == 14
    assert len(train2) == 148
    assert len(train0) == 0
    assert len(test1) == 1
    assert len(test2) == 17
    assert len(test0) == 0
