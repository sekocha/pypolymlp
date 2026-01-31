"""Tests of neighbor calculations."""

import glob
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

unitcell = Poscar(path_file + "poscars/POSCAR.perovskite.SrTiO3").structure


def test_eval1():
    """Test pair-type polymlp."""
    pot = path_file + "mlps/polymlp.lammps.pair.SrTiO3"
    prop = Properties(pot=pot)

    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.63203426437243, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-1.9204671, -1.9204671, -1.9204671, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval2_cond():
    """Test pair-type polymlp with conditional conditions."""
    pot = path_file + "mlps/polymlp.lammps.pair.cond.SrTiO3"
    prop = Properties(pot=pot)

    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.64306562035659, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-1.74745126, -1.74745126, -1.74745126, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval3():
    """Test gtinv-type polymlp."""
    pot = path_file + "mlps/polymlp.lammps.gtinv.SrTiO3"
    prop = Properties(pot=pot)

    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.64286166673613, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-0.02560392, -0.02560392, -0.02560392, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval4_cond():
    """Test gtinv-type polymlp with conditional conditions."""
    pot = path_file + "mlps/polymlp.lammps.gtinv.cond.SrTiO3"
    prop = Properties(pot=pot)

    energy_true = -31.642024569145274
    stresses_true = [0.21674893, 0.21674893, 0.21674893, 0, 0, 0]
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval_flexible1():
    """Test polymlp with flexible selection of pair features."""
    files = sorted(glob.glob(path_file + "mlps/polymlp.yaml.flexible.*.SrTiO3"))
    prop = Properties(pot=files)

    energy_true = -31.642657299368572
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)

    # stresses_true = [0.05689321, 0.05689321, 0.05689321, 0, 0, 0]
    # stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    # np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
