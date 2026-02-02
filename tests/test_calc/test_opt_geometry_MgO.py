"""Tests of geometry optimization in MgO."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_opt1():
    """Test optimization with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(path_file + "poscars/POSCAR.RS.MgO").structure
    opt = GeometryOptimization(
        pot=pot,
        cell=unitcell,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 21
    assert opt._basis_f.shape == (24, 21)
    assert opt._basis_axis is None

    opt.run()
    assert opt.energy == pytest.approx(-40.225125687168294, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces, 0.0, atol=1e-5)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()

    n_atom = len(unitcell.elements)
    xvec = np.random.random((n_atom - 1) * 3 + 6)
    x1, x2 = opt.split(xvec)
    assert len(x1) == (n_atom - 1) * 3
    assert len(x2) == 6


def test_opt2():
    """Test optimization with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(path_file + "poscars/POSCAR.RS.MgO").structure
    opt = GeometryOptimization(
        pot=pot,
        cell=unitcell,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 30
    assert opt._basis_f.shape == (24, 21)
    assert opt._basis_axis.shape == (9, 9)

    opt.run()
    assert opt.energy == pytest.approx(-40.225176328737426, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()

    n_atom = len(unitcell.elements)
    xvec = np.random.random((n_atom - 1) * 3 + 6)
    x1, x2 = opt.split(xvec)
    assert len(x1) == (n_atom - 1) * 3
    assert len(x2) == 6


def test_opt3():
    """Test optimization with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(path_file + "poscars/POSCAR.RS.idealMgO").structure
    opt = GeometryOptimization(
        pot=pot,
        cell=unitcell,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 1
    assert opt._basis_f is None
    assert opt._basis_axis.shape == (9, 1)

    opt.run()
    assert opt.energy == pytest.approx(-40.225176328737426, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()

    xvec = np.random.random(1)
    x1, x2 = opt.split(xvec)
    assert len(x1) == 0
    assert len(x2) == 1
