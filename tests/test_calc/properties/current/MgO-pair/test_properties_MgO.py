"""Tests of neighbor calculations."""

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

    energy_true = -40.22469744315832
    forces_true = [
        [-0.03962958, -0.01188776, -0.07928375],
        [-0.00459307, 0.00331611, 0.02210267],
        [0.01105381, -0.00137797, 0.022104],
        [0.01105096, 0.00331545, -0.00918516],
        [0.00978188, 0.00177061, 0.01180386],
        [0.00590284, 0.00293358, 0.01180553],
        [0.00589926, 0.00176979, 0.01958533],
        [0.0005339, 0.00016018, 0.00106752],
    ]
    stresses_true = [
        -0.17379186,
        -0.17521681,
        -0.16909401,
        0.00004465,
        0.00008926,
        0.00029751,
    ]

    assert energy == pytest.approx(energy_true, rel=1e-8)
    np.testing.assert_allclose(forces.T, forces_true, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)

    prop = Properties(pot=cwd / "polymlp.yaml.pair")
    energy, forces, stresses = prop.eval(unitcell)

    assert energy == pytest.approx(energy_true, rel=1e-8)
    np.testing.assert_allclose(forces.T, forces_true, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval2_yaml():
    unitcell = Poscar(cwd / "POSCAR").structure
    prop = Properties(pot=cwd / "polymlp.yaml.gtinv")
    energy, forces, stresses = prop.eval(unitcell)
    print(forces.T)

    energy_true = -40.223320043232334
    forces_true = [
        [-0.03882777, -0.0116472, -0.07768031],
        [-0.00302382, 0.00324809, 0.02165033],
        [0.01082712, -0.00090718, 0.02165147],
        [0.01082468, 0.00324753, -0.00604679],
        [0.00925154, 0.00182258, 0.01215086],
        [0.00607581, 0.00277448, 0.01215188],
        [0.00607362, 0.00182207, 0.0185246],
        [-0.00120118, -0.00036037, -0.00240204],
    ]
    stresses_true = [
        -2.78575100e-02,
        -2.91029804e-02,
        -2.37508486e-02,
        1.32744242e-05,
        2.65114887e-05,
        8.83338671e-05,
    ]

    assert energy == pytest.approx(energy_true, rel=1e-8)
    np.testing.assert_allclose(forces.T, forces_true, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
