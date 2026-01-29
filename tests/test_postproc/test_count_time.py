"""Tests of count time."""

from pathlib import Path

from pypolymlp.postproc.count_time import PolymlpCost

cwd = Path(__file__).parent


def test_load_mlp():
    """Test PolymlpCost."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    cost = PolymlpCost(pot=file1, supercell=(1, 1, 1))
    assert cost._elements == ["Ti"]
    cost.run(n_calc=1, write_yaml=False)


def test_load_mlp2():
    paths = [str(cwd) + "/../files/mlps_count_times/"]
    cost = PolymlpCost(path_pot=paths, supercell=(1, 1, 1))
    cost.run(n_calc=1, write_yaml=False)
