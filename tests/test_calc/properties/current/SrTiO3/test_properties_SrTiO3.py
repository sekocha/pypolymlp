"""Tests of neighbor calculations."""

import glob
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent


def test_eval1():
    unitcell = Poscar(cwd / "POSCAR").structure

    prop = Properties(pot=cwd / "polymlp.lammps.pair")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.63203426437243, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-1.9204671, -1.9204671, -1.9204671, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    prop = Properties(pot=cwd / "polymlp.lammps.pair.cond")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.64306562035659, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-1.74745126, -1.74745126, -1.74745126, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    prop = Properties(pot=cwd / "polymlp.lammps.gtinv")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-31.64286166673613, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses_true = [-0.02560392, -0.02560392, -0.02560392, 0, 0, 0]
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval_cond():
    unitcell = Poscar(cwd / "POSCAR").structure

    energy_true = -31.642024569145274
    stresses_true = [0.21674893, 0.21674893, 0.21674893, 0, 0, 0]

    prop = Properties(pot=cwd / "polymlp.lammps.gtinv.cond")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    prop = Properties(pot=cwd / "polymlp.yaml.gtinv.cond")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval_flexible1():
    unitcell = Poscar(cwd / "POSCAR").structure

    energy_true = -31.64262794659518
    stresses_true = [0.05689321, 0.05689321, 0.05689321, 0, 0, 0]

    files = sorted(glob.glob(str(cwd) + "/polymlp.lammps.flexible.*"))
    prop = Properties(pot=files)
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    files = sorted(glob.glob(str(cwd) + "/polymlp.yaml.flexible.*"))
    prop = Properties(pot=files)
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-6)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
