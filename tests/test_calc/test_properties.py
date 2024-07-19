"""Tests of neighbor calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent


def test_eval():
    unitcell = Poscar(cwd / "POSCAR").structure
    prop = Properties(pot=cwd / "polymlp.lammps")
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-10.114213158625798, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        stresses,
        [1.45790649e-01, 1.45790649e-01, 1.45790649e-01, 0.0, 0.0, 0.0],
        atol=1e-5,
    )


def test_eval2():
    prop = Properties(pot=cwd / "polymlp.lammps")

    unitcell = Poscar(cwd / "POSCAR2").structure
    energy, forces, stresses = prop.eval(unitcell)
    unitcell2 = Poscar(cwd / "POSCAR3").structure
    energy2, forces2, stresses2 = prop.eval(unitcell2)
    assert energy == energy2 == pytest.approx(-164.22234804626612, abs=1e-12)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    assert forces2[0][0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        stresses,
        [1.723271e-05, 1.723271e-05, 1.723271e-05, 0.0, 0.0, 0.0],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        stresses2,
        [1.723271e-05, 1.723271e-05, 1.723271e-05, 0.0, 0.0, 0.0],
        atol=1e-5,
    )
