"""Tests of neighbor calculations."""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "/mlps/polymlp.lammps.gtinv.Cu-Ag-Au"
polymlp = PypolymlpCalc(pot=pot)


def test_eval_neighbors1():
    """Test property calculations for different cell shapes of identical structure."""
    energy_true = -10.114213158625798
    for i in range(1, 26):
        filename = path_file + "poscars/POSCAR-" + str(i).zfill(3) + ".neighbor_check"
        polymlp.load_poscars(filename)
        energies, _, _ = polymlp.eval()
        assert energies[0] == pytest.approx(energy_true, rel=1e-12)


def test_eval_neighbors2():
    """Test property calculations for different cell shapes of identical structure."""
    energy_true = -164.2223480462661
    for i in range(101, 126):
        filename = path_file + "poscars/POSCAR-" + str(i).zfill(3) + ".neighbor_check"
        polymlp.load_poscars(filename)
        energies, _, _ = polymlp.eval()
        assert energies[0] == pytest.approx(energy_true, rel=1e-12)
