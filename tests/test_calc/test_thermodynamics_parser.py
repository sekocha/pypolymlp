"""Tests of thermodynamics_parser."""

import glob
from pathlib import Path

import pytest

from pypolymlp.calculator.thermodynamics.thermodynamics_parser import (
    _count_data_minimum_size,
    _count_data_size,
    _get_common_grid,
    _get_grid_data,
    load_electron_yamls,
    load_sscha_yamls,
    load_ti_yamls,
    load_yamls,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


files_sscha = sorted(glob.glob(path_file + "sscha/*.yaml"))
data_sscha = load_sscha_yamls(files_sscha)

files_el = glob.glob(path_file + "electrons/*.yaml")
data_el = load_electron_yamls(files_el)

files_ti = glob.glob(path_file + "ti/*/*/polymlp_ti.yaml")
data_ti = load_ti_yamls(files_ti)


def test_load_sscha_yamls():
    """Test load_sscha_yamls."""
    data = data_sscha
    assert len(data) == 165

    cvols, ctemps = _count_data_size(data)
    assert len(cvols) == 15
    assert len(ctemps) == 11
    assert ctemps[100] == 15


def test_load_electron_yamls():
    """Test load_electron_yamls."""
    data = data_el
    assert len(data) == 2415

    cvols, ctemps = _count_data_size(data)
    assert len(cvols) == 15
    assert len(ctemps) == 161
    assert ctemps[100] == 15


def test_load_ti_yamls():
    """Test load_ti_yamls."""
    data = data_ti
    assert len(data[0]) == 165
    assert len(data[1]) == 165

    cvols, ctemps = _count_data_size(data[0])
    assert len(cvols) == 15
    assert len(ctemps) == 11
    assert ctemps[100] == 15

    cvols, ctemps = _count_data_size(data[1])
    assert len(cvols) == 15
    assert len(ctemps) == 11
    assert ctemps[100] == 15


def test_count_data_minimum_size():
    """Test _count_data_minimum_size."""
    data_all = [data_sscha, data_el, data_ti[0]]
    cvols, ctemps = _count_data_minimum_size(data_all)

    assert len(cvols) == 15
    assert len(ctemps) == 11
    assert ctemps[100] == 15

    volumes, temperatures = _get_common_grid(data_all)
    assert len(volumes) == 15
    assert len(temperatures) == 11
    assert volumes[1] == pytest.approx(10.778)
    assert temperatures[1] == 100

    grid_sscha = _get_grid_data(data_sscha, volumes, temperatures)
    assert grid_sscha.shape == (15, 11)


def test_load_yamls():
    """Test load_yamls."""
    grids = load_yamls(
        yamls_sscha=files_sscha[:100],
        yamls_electron=files_el,
        yamls_ti=files_ti,
        n_require=5,
    )
    assert grids[0].shape == (9, 11)
    assert grids[1].shape == (9, 11)
    assert grids[2].shape == (9, 11)
    assert grids[3].shape == (9, 11)
    assert grids[4] is None
