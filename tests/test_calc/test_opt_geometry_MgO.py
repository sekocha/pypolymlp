"""Tests of geometry optimization in MgO."""

import copy
import os

import numpy as np
import pytest

from pypolymlp.calculator.opt_geometry import GeometryOptimization


def test_opt1(unitcell_disp_pair_MgO):
    """Test optimization with pair polymlp in MgO."""
    unitcell1, pot, prop = unitcell_disp_pair_MgO
    unitcell = copy.deepcopy(unitcell1)
    opt = GeometryOptimization(
        properties=prop,
        cell=unitcell,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 21
    assert opt._basis_f.shape == (24, 21)
    assert opt._basis_a is None

    opt.run()
    assert opt.energy == pytest.approx(-40.225125687168294, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces, 0.0, atol=1e-5)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()


def test_opt2(unitcell_disp_pair_MgO):
    """Test optimization with pair polymlp in MgO."""
    unitcell1, pot, prop = unitcell_disp_pair_MgO
    unitcell = copy.deepcopy(unitcell1)
    opt = GeometryOptimization(
        properties=prop,
        cell=unitcell,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 30
    assert opt._basis_f.shape == (24, 21)
    assert opt._basis_a.shape == (9, 9)

    opt.run()
    assert opt.energy == pytest.approx(-40.225176328737426, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()


def test_opt3(unitcell_pair_MgO):
    """Test optimization with pair polymlp in MgO."""
    unitcell1, pot, prop = unitcell_pair_MgO
    unitcell = copy.deepcopy(unitcell1)
    opt = GeometryOptimization(
        properties=prop,
        cell=unitcell,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
        verbose=True,
    )
    assert opt._x0.shape[0] == 1
    assert opt._basis_f is None
    assert opt._basis_a.shape == (9, 1)

    opt.run()
    assert opt.energy == pytest.approx(-40.225176328737426, rel=1e-6)
    assert opt.success
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(opt.residual_forces[0], 0.0, atol=1e-3)
    opt.write_poscar(filename="tmp")
    os.remove("tmp")
    opt.print_structure()
