"""Tests of api_thermodynamics."""

import copy
from pathlib import Path

import pytest

from pypolymlp.calculator.thermodynamics.api_thermodynamics import (
    calculate_reference_grid,
    compute_grid_sum,
    load_thermodynamics_yaml,
    set_reference_paths,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_set_reference_paths(thermodynamics_grids_Cu):
    """Test set_reference_paths."""
    _, _, grid_ti, _, _ = thermodynamics_grids_Cu
    grid_ti = set_reference_paths(grid_ti)
    for i, j, d in grid_ti:
        fc2hdf5 = "/".join(d.path_yaml.split("/")[:-1]) + "/fc2.hdf5"
        assert d.path_fc2 == fc2hdf5


def test_calculate_reference_grid(thermodynamics_grids_Cu):
    """Test calculate_reference_grid."""
    grid_sscha, _, grid_ti, _, _ = thermodynamics_grids_Cu
    grid_ti = set_reference_paths(grid_ti)
    grid_ti.copy_static_data(grid_sscha)

    tmp = copy.deepcopy(grid_ti)
    for i, j, d in tmp:
        if i > 0:
            d.volume = None
    grid_ref = calculate_reference_grid(tmp)
    assert grid_ref.shape == grid_ti.shape
    assert grid_ref[0, 5].free_energy == pytest.approx(-4.117530994299927)


def test_compute_grid_sum(thermodynamics_grids_Cu):
    """Test compute_grid_sum."""
    grid1, grid2, _, _, _ = thermodynamics_grids_Cu
    grid_s = compute_grid_sum([grid1, grid2])
    assert grid_s[1, 2].free_energy == pytest.approx(-4.028855095413766)
    assert grid_s[1, 2].entropy == pytest.approx(0.00020649367346735925)


def test_load_thermodynamics_yaml():
    """Test load_thermodynamics_yaml."""
    data = load_thermodynamics_yaml(path_file + "sscha.yaml")
    assert len(data.temperatures) == 16
    assert len(data.eq_volumes) == 16
    assert len(data.bm) == 16
    assert len(data.eq_helmholtz) == 16
    assert len(data.eq_entropy) == 16
    assert len(data.eq_cp) == 16
    assert len(data.eos_data) == 16
    assert len(data.eos_fit_data) == 16
    assert len(data.gibbs) == 16

    data1 = data.get_T_F()
    assert data1.shape == (16, 2)
