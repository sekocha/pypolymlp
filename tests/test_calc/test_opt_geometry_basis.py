"""Tests of basis sets for geometry optimization."""

from pathlib import Path

import pytest

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.symfc_utils import construct_basis_cartesian

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_basis_SrTiO3_tetra():
    """Test construct_basis_cell and construct_basis_cartesian."""
    poscar = path_file + "poscars/POSCAR.tetra.SrTiO3"
    st = Poscar(poscar).structure
    basis, st_rev = construct_basis_cell(st, verbose=True)
    assert basis[0][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis[4][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis[8][1] == pytest.approx(1.0, rel=1e-8)

    basis = construct_basis_cartesian(st_rev)
    assert basis.shape[1] == 1


def test_basis_ZnS_wurtzite():
    """Test construct_basis_cell and construct_basis_cartesian."""
    poscar = path_file + "poscars/POSCAR.WZ.ZnS"
    st = Poscar(poscar).structure
    basis, st_rev = construct_basis_cell(st, verbose=True)
    assert basis[0][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis[1][0] == pytest.approx(-0.35355339, rel=1e-5)
    assert basis[4][0] == pytest.approx(0.61237244, rel=1e-5)
    assert basis[8][1] == pytest.approx(1.0, rel=1e-8)

    basis = construct_basis_cartesian(st_rev)
    assert abs(basis[2][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis[5][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis[8][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis[11][0]) == pytest.approx(0.5, rel=1e-5)
    assert basis[2][0] == -basis[8][0]
    assert basis.shape[1] == 1
