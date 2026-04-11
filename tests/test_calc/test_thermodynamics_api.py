"""Tests of api_thermodynamics."""

import copy
import os
import shutil
from pathlib import Path

import pytest

from pypolymlp.calculator.thermodynamics.api_thermodynamics import (
    Thermodynamics,
    ThermodynamicsData,
    calculate_reference_grid,
    compute_grid_sum,
    load_thermodynamics_yaml,
    set_reference_paths,
)
from pypolymlp.calculator.thermodynamics.thermodynamics_grid import GridVT

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_Thermodynamics(thermodynamics_grids_Cu):
    """Test Thermodynamics."""
    grid, _, _, _, _ = thermodynamics_grids_Cu
    thermo = Thermodynamics(grid)
    thermo.fit_free_energy_volume()
    thermo.fit_entropy_volume()

    free_energies = thermo.eval_free_energy_equilibrium()
    entropies = thermo.eval_entropy_equilibrium()
    cp = thermo.eval_cp_numerical()

    assert free_energies[5] == pytest.approx(-4.1612396567984415)
    assert entropies[5] == pytest.approx(0.00045619918968161136)
    assert cp[5] == pytest.approx(24.94523630348574)

    thermo.save_thermodynamics_yaml(filename="tmp.yaml")
    os.remove("tmp.yaml")

    volumes = [18, 20, 22]
    free_energies = thermo.eval_free_energies(volumes)
    entropies = thermo.eval_entropies(volumes)
    gibbs = thermo.eval_gibbs_free_energies(volumes)
    assert free_energies.shape == (3, 11)
    assert entropies.shape == (3, 11)
    assert gibbs.shape == (11, 3, 2)

    assert free_energies[1, 5] == pytest.approx(-3.3379014999686087)
    assert entropies[1, 5] == pytest.approx(0.0007898582946575324)

    assert len(thermo.volumes) == 15
    assert len(thermo.temperatures) == 11
    assert thermo.data.shape == (15, 11)
    assert isinstance(thermo.grid, GridVT)

    assert len(thermo.fitted_models.extract(2)) == 3


def test_ThermodynamicsData(thermodynamics_grids_Cu):
    """Test ThermodynamicsData."""
    grid_sscha, grid_el, _, _, _ = thermodynamics_grids_Cu
    grid_sscha_el = compute_grid_sum([grid_sscha, grid_el])

    data = ThermodynamicsData(
        sscha=Thermodynamics(grid_sscha),
        sscha_el=Thermodynamics(grid_sscha_el),
    )
    data.run()
    data.save(path="tmp")
    shutil.rmtree("tmp")


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
