"""Tests of neighbor calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell

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


def test_eval_neighbors_MgO():
    """Test property calculations for different cell shapes of identical structure."""
    pot = path_file + "/mlps/polymlp.yaml.pair.MgO"
    polymlp = PypolymlpCalc(pot=pot)

    unitcell = Poscar(path_file + "poscars/POSCAR.RS.idealMgO").structure
    expansions = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [-1, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-1, 1, 0], [-1, -1, 1]],
        [[1, 0, 0], [-3, 1, 0], [3, 3, 1]],
        [[1, 0, 0], [3, 1, 0], [3, 3, 1]],
        [[1, 0, 0], [-3, 1, 0], [-3, -3, 1]],
        [[1, 0, 0], [5, 1, 0], [5, 5, 1]],
        [[1, 0, 0], [-5, 1, 0], [-5, -5, 1]],
        [[1, 0, 0], [10, 1, 0], [10, 10, 1]],
        [[1, 0, 0], [-10, 1, 0], [-10, -10, 1]],
        [[1, 3, 3], [0, 1, 3], [0, 0, 1]],
        [[1, 4, 4], [0, 1, 4], [0, 0, 1]],
        [[1, 5, 5], [0, 1, 5], [0, 0, 1]],
        [[1, -5, -5], [0, 1, -5], [0, 0, 1]],
        [[1, 10, 10], [0, 1, 10], [0, 0, 1]],
        [[1, -10, -10], [0, 1, -10], [0, 0, 1]],
        [[1, 0, 0], [5, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-5, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [10, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [-10, 1, 0], [0, 0, 1]],
    ]

    structures = [supercell(unitcell, np.array(hnf)) for hnf in expansions]
    energies, forces, stresses = polymlp.eval(structures)
    np.testing.assert_allclose(energies, -40.225125687168706, atol=1e-12)
