"""Test PypolymlpCalcProperties."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.api_calculator import PypolymlpCalcProperties

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_polymlp_Al():
    """Test PypolymlpCalcProperties with polymlp."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    e, f, s = polymlp.eval(unitcell)
    assert e == pytest.approx(-13.711192266235127)
    assert f.shape == (3, 4)
    assert s.shape == (6,)

    unitcells = polymlp.load_poscars([poscar, poscar])
    e, f, s = polymlp.eval_multiple(unitcells)
    np.testing.assert_allclose(e, [-13.711192266235127, -13.711192266235127])
    assert np.array(f).shape == (2, 3, 4)
    assert s.shape == (2, 6)

    assert polymlp.calculator is not None
    assert polymlp.static_calculator is not None
    assert polymlp.dynamic_calculator is None
    assert tuple(polymlp.elements) == ("Al",)
    assert polymlp.pot.split("/")[-1] == "polymlp.yaml.gtinv.Al"
    assert polymlp.energies.shape == (2,)
    assert np.array(polymlp.forces).shape == (2, 3, 4)
    assert polymlp.stresses.shape == (2, 6)
    assert polymlp.stresses_gpa.shape == (2, 6)
    assert polymlp.stresses_gpa[0, 0] == pytest.approx(0.12686092678098726)


def test_sscha_Al():
    """Test PypolymlpCalcProperties with polymlp and SSCHA."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=700,
        tol=0.01,
        mixing=0.5,
        path="tmp",
        use_mkl=False,
    )
    e, f, s = polymlp.eval(unitcell)

    assert polymlp.calculator is not None
    assert polymlp.static_calculator is not None
    assert polymlp.dynamic_calculator is not None
    assert tuple(polymlp.elements) == ("Al",)
    assert polymlp.pot.split("/")[-1] == "polymlp.yaml.gtinv.Al"

    assert polymlp.sscha is not None
    assert polymlp.force_constants.shape == (32, 32, 3, 3)

    assert polymlp.energies is None
    assert polymlp.forces is None
    assert polymlp.stresses is None
    assert polymlp.stresses_gpa is None

    shutil.rmtree("tmp")


def test_lammps():
    """Test PypolymlpCalcProperties with lammps."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    pytest.importorskip("lammps")
    polymlp.set_lammps(elements=("Al",), pot=pot)

    unitcell = polymlp.load_poscar(poscar)
    e, f, s = polymlp.eval(unitcell)
    assert e == pytest.approx(-13.711192266235127)
    assert f.shape == (3, 4)
    assert s.shape == (6,)


def test_lammps_sscha():
    """Test PypolymlpCalcProperties with lammps and SSCHA."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    pytest.importorskip("lammps")
    polymlp.set_lammps(elements=("Al",), pot=pot)

    unitcell = polymlp.load_poscar(poscar)
    polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=700,
        tol=0.01,
        mixing=0.5,
        path="tmp",
        use_mkl=False,
    )
    e, f, s = polymlp.eval(unitcell)
    assert f.shape == (3, 4)
    assert s.shape == (6,)
    shutil.rmtree("tmp")
