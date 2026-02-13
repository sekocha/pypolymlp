"""Tests of structure optimization using API"""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_stropt_SrTiO3_tetra():
    """Test geometry optimization using API in tetragonal SrTiO3."""
    poscar = path_file + "poscars/POSCAR.tetra.SrTiO3"
    pot = path_file + "mlps/polymlp.lammps.gtinv.SrTiO3"

    polymlp = PypolymlpCalc(pot=pot)
    polymlp.load_poscars(poscar)
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization(gtol=1e-5, method="BFGS")
    e0, n_iter, success = polymlp.go_data

    assert e0 == pytest.approx(-126.6061605040393, rel=1e-6)
    np.testing.assert_allclose(polymlp._go.residual_forces[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(polymlp._go.residual_forces[1], 0.0, atol=1e-5)

    e, f, s = polymlp.eval(polymlp.first_structure)
    assert e[0] == pytest.approx(-126.6061605040393, rel=1e-6)
    np.testing.assert_allclose(f, 0.0, atol=1e-4)
    np.testing.assert_allclose(s, 0.0, atol=1e-4)


def test_stropt_ZnS_wurtzite1():
    """Test geometry optimization using API in wurtzite ZnS."""
    poscar = path_file + "poscars/POSCAR.WZ.ZnS"
    pot = path_file + "mlps/polymlp.lammps.gtinv.ZnS"
    polymlp = PypolymlpCalc(pot=pot)
    polymlp.load_poscars(poscar)

    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization(gtol=1e-5, method="CG")
    e0, n_iter, success = polymlp.go_data

    e_ref = -13.136821752018669
    assert e0 == pytest.approx(e_ref, rel=1e-6)
    np.testing.assert_allclose(polymlp._go.residual_forces[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(polymlp._go.residual_forces[1], 0.0, atol=1e-5)

    e, f, s = polymlp.eval(polymlp.first_structure)
    assert e0 == pytest.approx(e_ref, rel=1e-6)
    np.testing.assert_allclose(f, 0.0, atol=1e-4)
    np.testing.assert_allclose(s, 0.0, atol=1e-4)


def test_stropt_ZnS_wurtzite2():
    """Test geometry optimization using API in wurtzite ZnS."""
    poscar = path_file + "poscars/POSCAR.WZ.ZnS"
    pot = path_file + "mlps/polymlp.lammps.gtinv.ZnS"
    polymlp = PypolymlpCalc(pot=pot)
    polymlp.load_poscars(poscar)
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=True,
        relax_volume=False,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization(gtol=1e-5, method="BFGS")
    e0, n_iter, success = polymlp.go_data

    e_ref = -13.09056737444269
    assert e0 == pytest.approx(e_ref, rel=1e-5)


def test_stropt_ZnS_wurtzite3():
    """Test geometry optimization using API in wurtzite ZnS."""
    poscar = path_file + "poscars/POSCAR.WZ.ZnS"
    pot = path_file + "mlps/polymlp.lammps.gtinv.ZnS"
    polymlp = PypolymlpCalc(pot=pot)
    polymlp.load_poscars(poscar)
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization(gtol=1e-5, method="BFGS")
    e0, n_iter, success = polymlp.go_data

    e_ref = -13.031440230712214
    assert e0 == pytest.approx(e_ref, rel=1e-6)
