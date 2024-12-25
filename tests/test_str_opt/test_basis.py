"""Tests of basis sets using API"""

from pathlib import Path

import pytest

# symfc and spglib are required.
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.symfc_utils import construct_basis_cartesian

cwd = Path(__file__).parent


def test_basis_SrTiO3_tetra():
    poscar = str(cwd) + "/data-SrTiO3-tetra/POSCAR"
    st = Poscar(poscar).structure
    basis, st_rev = construct_basis_cell(st, verbose=True)
    assert basis[0][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis[4][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis[8][1] == pytest.approx(1.0, rel=1e-8)

    basis = construct_basis_cartesian(st_rev)
    assert basis.shape[1] == 1


def test_basis_ZnS_wurtzite():
    poscar = str(cwd) + "/data-ZnS-wurtzite/POSCAR"
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
