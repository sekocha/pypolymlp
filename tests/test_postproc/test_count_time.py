"""Tests of count time."""

from pathlib import Path

import numpy as np

from pypolymlp.postproc.count_time import PolymlpCost

cwd = Path(__file__).parent


def test_load_mlp():
    """Test PolymlpCost."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    cost = PolymlpCost(pot=file1, supercell_matrix=(1, 1, 1))
    assert cost._elements == ["Ti"]
    cost.run(n_calc=1, write_yaml=False)


def test_load_mlp1():
    """Test PolymlpCost."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    mat = np.array([[2, 0, 0], [0, 2, 0], [1, 0, 2]])
    cost = PolymlpCost(pot=file1, supercell_matrix=mat)
    assert cost._elements == ["Ti"]
    cost.run(n_calc=1, write_yaml=False)


def test_load_mlp2():
    paths = [str(cwd) + "/../files/mlps_count_times/"]
    cost = PolymlpCost(path_pot=paths, supercell_matrix=(1, 1, 1))
    cost.run(n_calc=1, write_yaml=False)
