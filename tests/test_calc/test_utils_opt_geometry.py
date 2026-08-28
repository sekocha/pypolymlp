"""Tests of basis sets for geometry optimization."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.utils.opt_geometry_utils import BasisSetGO
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.WZ.ZnS"
st_ZnS = Poscar(poscar).structure


def test_BasisSetGO_ZnS_wurtzite1():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a[0][0] == pytest.approx(0.70710678, rel=1e-5)
    assert basis_a[1][0] == pytest.approx(-0.35355339, rel=1e-5)
    assert basis_a[4][0] == pytest.approx(0.61237244, rel=1e-5)
    assert basis_a[8][1] == pytest.approx(1.0, rel=1e-8)
    assert basis_a.shape == (9, 2)

    assert abs(basis_f[2][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis_f[5][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis_f[8][0]) == pytest.approx(0.5, rel=1e-5)
    assert abs(basis_f[11][0]) == pytest.approx(0.5, rel=1e-5)
    assert basis_f[2][0] == -basis_f[8][0]
    assert basis_f.shape == (12, 1)

    assert len(basis.init_structure.elements) == 4
    assert len(basis.init_coeffs) == 3
    a = basis.axis(np.ones(2))
    assert a.shape == (3, 3)
    pos = basis.positions(np.ones(1))
    assert pos.shape == (3, 4)
    st = basis.structure(np.ones(3))
    assert len(st.elements) == 4
    c1, c2 = basis.split(np.ones(3))
    assert len(c1) == 1
    assert len(c2) == 2


def test_BasisSetGO_ZnS_wurtzite2():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=True,
        relax_positions=False,
        with_sym=True,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 2)
    assert basis_f is None


def test_BasisSetGO_ZnS_wurtzite3():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
        with_sym=True,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a is None
    assert basis_f.shape == (12, 1)


def test_BasisSetGO_ZnS_wurtzite4():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=False,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 1)
    assert basis_f.shape == (12, 1)


def test_BasisSetGO_ZnS_wurtzite5():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=False,
        relax_positions=True,
        with_sym=True,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 2)
    assert basis_f.shape == (12, 1)


def test_BasisSetGO_ZnS_wurtzite_nosym1():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=False,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 9)
    assert basis_f.shape == (12, 12)


def test_BasisSetGO_ZnS_wurtzite_nosym2():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=True,
        relax_positions=False,
        with_sym=False,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 9)
    assert basis_f is None


def test_BasisSetGO_ZnS_wurtzite_nosym3():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
        with_sym=False,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a is None
    assert basis_f.shape == (12, 12)


def test_BasisSetGO_ZnS_wurtzite_nosym4():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=False,
        relax_volume=True,
        relax_positions=True,
        with_sym=False,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 1)
    assert basis_f.shape == (12, 12)


def test_BasisSetGO_ZnS_wurtzite_nosym5():
    """Test BasisSetGO."""
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=False,
        relax_positions=True,
        with_sym=False,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 9)
    assert basis_f.shape == (12, 12)


def test_BasisSetGO_ZnS_wurtzite_run():
    """Test BasisSetGO."""
    bool2 = [True, False]
    boolprods = itertools.product(*[bool2, bool2, bool2, bool2])
    for with_sym, relax_cell, relax_volume, relax_positions in boolprods:
        try:
            _ = BasisSetGO(
                st_ZnS,
                elements=("Zn", "S"),
                with_sym=with_sym,
                relax_cell=relax_cell,
                relax_volume=relax_volume,
                relax_positions=relax_positions,
                verbose=True,
            )
        except RuntimeError:
            pass


def test_BasisSetGO_ZnS_wurtzite_sd1():
    """Test BasisSetGO."""
    sd_cell = np.ones((3, 3), dtype=bool)
    sd_cell[0, 0] = False
    sd_pos = np.ones((3, 4), dtype=bool)
    sd_pos[2, 0] = False
    basis = BasisSetGO(
        cell=st_ZnS,
        elements=("Zn", "S"),
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        with_sym=True,
        selective_dynamics_cell=sd_cell,
        selective_dynamics_positions=sd_pos,
    )
    basis_a, basis_f = basis.basis_a, basis.basis_f
    assert basis_a.shape == (9, 1)
    assert basis_f is None
