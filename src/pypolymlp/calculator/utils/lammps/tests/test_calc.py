"""Tests of lammps-python calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent


def test_eval1():
    """Test lammps API."""
    path = str(cwd) + "/Ti-Al/"
    pot = path + "/polymlp.lammps"
    elements = ["Ti", "Al"]
    prop = PropertiesLammps(
        elements=elements,
        pot=pot,
        style="polymlp",
        style_command="pair_style",
        coeff_command="pair_coeff",
        log=False,
        verbose=False,
    )

    st = Poscar(path + "POSCAR-Al").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-6.650991070412197, abs=1e-12)

    st = Poscar(path + "POSCAR-Al2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-6.650991070412197, abs=1e-12)

    st = Poscar(path + "POSCAR-Ti").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-10.413251553867973, abs=1e-12)

    st = Poscar(path + "POSCAR-Ti2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-10.413251553867973, abs=1e-12)

    st = Poscar(path + "POSCAR-TiAl-2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-9.211925132828384, abs=1e-12)

    st = Poscar(path + "POSCAR-AlTi").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-7.591693394073458, abs=1e-12)
    f_true = [[-0.35067, 0.0, 0.0], [-2.66416, 0.0, 0.0], [3.01483, 0.0, 0.0]]
    np.testing.assert_allclose(f0.T, f_true, atol=1e-4)

    st = Poscar(path + "POSCAR-TiAl").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-7.591693394073495, abs=1e-12)
    f_true = [[-2.66416, 0.0, 0.0], [3.01483, 0.0, 0.0], [-0.35067, 0.0, 0.0]]
    np.testing.assert_allclose(f0.T, f_true, atol=1e-4)


def test_eval_eam():
    """Test lammps API using EAM potential."""
    path = str(cwd) + "/Ti-Al/"
    pot = path + "Zope-Ti-Al-2003.eam.alloy"
    elements = ["Ti", "Al"]

    prop = PropertiesLammps(
        elements=elements,
        pot=pot,
        style="eam/alloy",
        style_command="pair_style",
        coeff_command="pair_coeff",
        log=False,
        verbose=False,
    )
    st = Poscar(path + "POSCAR-Al").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-6.52596651890101, abs=1e-12)

    st = Poscar(path + "POSCAR-Al2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-6.52596651890101, abs=1e-12)

    st = Poscar(path + "POSCAR-Ti").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-9.608472280865415, abs=1e-12)

    st = Poscar(path + "POSCAR-Ti2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-9.608472280865415, abs=1e-12)

    st = Poscar(path + "POSCAR-TiAl-2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-8.78154896947258, abs=1e-12)

    st = Poscar(path + "POSCAR-AlTi").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(7.624104962299699, abs=1e-12)
    f_true = [[0.0638642, 0.0, 0.0], [-6.331095, 0.0, 0.0], [6.267231, 0.0, 0.0]]
    np.testing.assert_allclose(f0.T, f_true, atol=1e-4)

    st = Poscar(path + "POSCAR-TiAl").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(7.624104962299699, abs=1e-12)
    f_true = [[-6.331095, 0.0, 0.0], [6.267231, 0.0, 0.0], [0.0638642, 0.0, 0.0]]
    np.testing.assert_allclose(f0.T, f_true, atol=1e-4)


def test_eval2():
    path = str(cwd) + "/SrTiO3/"
    pot = path + "polymlp.lammps"

    elements = ["Sr", "Ti", "O"]
    prop = PropertiesLammps(
        elements=elements,
        pot=pot,
        style="polymlp",
        style_command="pair_style",
        coeff_command="pair_coeff",
        log=False,
        verbose=False,
    )

    st = Poscar(path + "POSCAR").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-31.642024569145264, abs=1e-12)

    st = Poscar(path + "POSCAR2").structure
    e0, f0, s0 = prop.eval(st)
    assert e0 == pytest.approx(-31.480951115669793, abs=1e-12)
    f_true = [
        [0.06260702, -0.13276562, 0.08225099],
        [-0.03023256, 0.11111077, 0.07090701],
        [-0.01750507, 0.01422164, -0.93302199],
        [-0.00449066, -0.01362461, 0.38731617],
        [-0.01037873, 0.02105782, 0.39254782],
    ]
    np.testing.assert_allclose(f0.T, f_true, atol=1e-4)
