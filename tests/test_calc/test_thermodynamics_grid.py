"""Tests of thermodynamics_grid."""

from pathlib import Path

import pytest

from pypolymlp.calculator.thermodynamics.thermodynamics_grid import (
    GridPointData,
    sum_grids,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_GridPointData1():
    """Test GridPointData."""
    data1 = GridPointData(
        volume=10,
        temperature=100,
        free_energy=1,
        entropy=1,
        energy=2,
        static_potential=-100,
        path_fc2="fc2.hdf5",
    )
    data2 = GridPointData(
        volume=10,
        temperature=100,
        free_energy=1,
        entropy=1,
        energy=2,
        static_potential=None,
        path_fc2=None,
    )
    data1.add(data2)
    assert data1.free_energy == 2
    assert data1.entropy == 2
    assert data1.energy == 4
    assert data1.static_potential is None
    assert data1.exist_attr("free_energy")
    assert not data1.exist_attr("heat_capacity")
    assert not data1.is_empty
    assert data1.unitcell is None
    assert data1.supercell_matrix is None


def test_GridPointData2(thermodynamics_grids_Cu):
    grid, _, _, _, _ = thermodynamics_grids_Cu
    data1 = grid[5, 5]
    assert data1.restart is not None
    assert data1.unitcell.axis.shape == (3, 3)
    assert data1.unitcell.positions.shape == (3, 4)
    assert data1.supercell_matrix.shape == (3, 3)


def test_GridVT(thermodynamics_grids_Cu):
    """Test GridVT."""
    grid, _, _, _, _ = thermodynamics_grids_Cu
    assert len(grid.volumes) == 15
    assert len(grid.temperatures) == 11
    assert grid.data.shape == (15, 11)
    assert grid.shape == (15, 11)

    assert len([i for i, j, d in grid]) == 165
    item = grid[3, 5]
    assert isinstance(item, GridPointData)
    grid[3, 5] = item

    grid.copy_static_data(grid)
    entropy = grid.get_properties("entropy")
    assert entropy.shape == (15, 11)

    zip1 = grid.get_volumes_properties("entropy")
    for temp, volumes, props in zip1:
        assert len(volumes) == 15
        assert len(props) == 15

    zip1 = grid.get_temperatures_properties("entropy")
    for volume, temperatures, props, _ in zip1:
        assert len(temperatures) == 11
        assert len(props) == 11

    fits = grid.fit_free_energy_volume()
    assert len(fits) == 11
    fits = grid.fit_entropy_volume()
    assert len(fits) == 11
    grid.print_predictions(fits[0], [10, 11], [1, 2])


def test_sum_grids(thermodynamics_grids_Cu):
    """Test sum_grids."""
    grid1, grid2, _, _, _ = thermodynamics_grids_Cu
    grid_s = sum_grids([grid1, grid2])
    assert grid_s[1, 2].free_energy == pytest.approx(-4.028855095413766)
    assert grid_s[1, 2].entropy == pytest.approx(0.00020649367346735925)
