"""Tests of calculations for Cu-Ag-Au."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "/mlps/polymlp.lammps.gtinv.Cu-Ag-Au"
prop = Properties(pot=pot)


def test_eval1():
    """Test properties for a standard calculation."""
    poscar = path_file + "/poscars/POSCAR1.Cu-Ag-Au"

    unitcell = Poscar(poscar).structure
    energy, forces, stresses = prop.eval(unitcell)

    assert energy == pytest.approx(-10.114213158625798, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [1.45790649e-01, 1.45790649e-01, 1.45790649e-01, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval2():
    """Test properties of largely-distorted binary structures."""
    poscar1 = path_file + "/poscars/POSCAR2.Cu-Ag-Au"
    poscar2 = path_file + "/poscars/POSCAR3.Cu-Ag-Au"

    unitcell1 = Poscar(poscar1).structure
    unitcell2 = Poscar(poscar2).structure

    energy, forces, stresses = prop.eval(unitcell1)
    energy2, forces2, stresses2 = prop.eval(unitcell2)

    assert energy == pytest.approx(-164.22234804626612, abs=1e-12)
    assert energy2 == pytest.approx(-164.22234804626612, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    assert forces2[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [1.723271e-05, 1.723271e-05, 1.723271e-05, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
    np.testing.assert_allclose(stresses2, stresses_true, atol=1e-5)


def test_eval3():
    """Test properties of largely-distorted ternary structures."""
    poscar = path_file + "/poscars/POSCAR4.Cu-Ag-Au"
    unitcell = Poscar(poscar).structure

    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-156.20431533922547, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0053143900922453646, abs=1e-12)
    stresses_true = [
        -1.41413638e00,
        -1.28681668e00,
        -1.37342653e00,
        5.42957694e-02,
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    poscar = path_file + "/poscars/POSCAR5.Cu-Ag-Au"
    unitcell = Poscar(poscar).structure
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-156.20431533922547, abs=1e-12)


def test_eval_multiple1():
    """Test properties of multiple structures using eval_multiple."""
    poscar1 = path_file + "/poscars/POSCAR1.Cu-Ag-Au"
    poscar2 = path_file + "/poscars/POSCAR2.Cu-Ag-Au"
    poscar3 = path_file + "/poscars/POSCAR3.Cu-Ag-Au"
    unitcell1 = Poscar(poscar1).structure
    unitcell2 = Poscar(poscar2).structure
    unitcell3 = Poscar(poscar3).structure

    energies, forces, stresses = prop.eval_multiple([unitcell1, unitcell2, unitcell3])

    assert energies[0] == pytest.approx(-10.114213158625798, abs=1e-12)
    assert energies[1] == pytest.approx(-164.22234804626612, abs=1e-12)
    assert energies[2] == pytest.approx(-164.22234804626612, abs=1e-12)
    assert forces[0][0][0] == pytest.approx(0.0, abs=1e-12)
    assert forces[1][0][0] == pytest.approx(0.0, abs=1e-12)
    assert forces[2][0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true1 = [1.45790649e-01, 1.45790649e-01, 1.45790649e-01, 0.0, 0.0, 0.0]
    stresses_true2 = [1.723271e-05, 1.723271e-05, 1.723271e-05, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(stresses[0], stresses_true1, atol=1e-5)
    np.testing.assert_allclose(stresses[1], stresses_true2, atol=1e-5)
    np.testing.assert_allclose(stresses[2], stresses_true2, atol=1e-5)
