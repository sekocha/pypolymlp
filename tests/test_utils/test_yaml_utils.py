"""Tests of yaml utility functions."""

from pathlib import Path

import yaml

from pypolymlp.utils.yaml_utils import load_cell, load_cells

cwd = Path(__file__).parent


def test_load_cell():
    """Test for load_cell and load_cells functions."""
    filename = str(cwd) + "/../files/sscha_results_0001.yaml"
    cell = load_cell(filename=filename)
    assert cell.positions.shape == (3, 40)

    unitcell, supercell = load_cells(filename=filename)
    assert unitcell.positions.shape == (3, 40)
    assert supercell.positions.shape == (3, 40)

    yaml_data = yaml.safe_load(open(filename))
    cell = load_cell(yaml_data=yaml_data)
    assert cell.positions.shape == (3, 40)

    unitcell, supercell = load_cells(yaml_data=yaml_data)
    assert unitcell.positions.shape == (3, 40)
    assert supercell.positions.shape == (3, 40)


# TODO
def test_load_data():
    """Test for load_data."""
    pass
