"""Tests of neighbor calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent


def test_eval1():
    unitcell = Poscar(cwd / "Ag-pair/POSCAR").structure
    prop = Properties(pot=cwd / "Ag-pair/mlp.lammps")
    energy, forces, stresses = prop.eval(unitcell)

    assert energy == pytest.approx(-10.031307591610425, abs=1e-12)
    forces_true = [
        [-1.24702646e-02, 9.72028834e-05, 6.18963049e-03, 6.18343122e-03],
        [-3.74121702e-03, 1.85699851e-03, 2.89620474e-05, 1.85525646e-03],
        [-2.49438077e-02, 1.23668884e-02, 1.23710083e-02, 2.05911033e-04],
    ]
    np.testing.assert_allclose(forces, forces_true, atol=1e-6)
    stresses_true = [
        -2.26126561e00,
        -2.26131407e00,
        -2.26107121e00,
        5.71350861e-05,
        1.22477289e-04,
        4.07360575e-04,
    ]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
