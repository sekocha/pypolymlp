"""Tests of vasp parser."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.interface_vasp import (
    check_vasprun_type,
    parse_energy_volume,
    parse_properties_from_vaspruns,
    parse_structures_from_poscars,
    parse_structures_from_vaspruns,
)

cwd = Path(__file__).parent


def test_functions_from_vaspruns():
    """Test for parse_properties_from_vaspruns and other functions."""
    vaspruns = [
        cwd / "./../files/vasprun-00001-MgO.xml",
        cwd / "./../files/vasprun-00002-MgO.xml",
    ]
    strs, (energies, forces, stresses) = parse_properties_from_vaspruns(vaspruns)
    assert forces[0].shape == (3, 64)
    assert forces[1].shape == (3, 64)
    assert forces[0][0, 3] == pytest.approx(0.00875877)
    assert forces[1][2, 5] == pytest.approx(0.00594728)

    stresses_true = [
        [
            [0.20440833, -0.00616032, -0.02821958],
            [-0.00616039, 0.21007008, 0.00932128],
            [-0.02821948, 0.00932117, 0.25478931],
        ],
        [
            [0.31962655, -0.08733854, 0.00825778],
            [-0.08733852, 0.11339135, 0.0465711],
            [0.00825789, 0.04657095, 0.23225083],
        ],
    ]
    np.testing.assert_allclose(energies, [-381.0917812, -381.09122093], atol=1e-6)
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-6)

    strs = parse_structures_from_vaspruns(vaspruns)
    assert len(strs) == 2
    assert [check_vasprun_type(v)[0] for v in vaspruns] == [False, False]

    ev = parse_energy_volume(vaspruns)
    np.testing.assert_allclose(
        ev, [[612.13782226, -381.0917812], [612.13394808, -381.09122093]], atol=1e-6
    )


def test_functions_from_poscars():
    """Test for parse_structures_from_poscars and other functions."""
    poscars = [
        cwd / "./../files/POSCAR-rocksalt",
        cwd / "./../files/POSCAR-rocksalt",
    ]
    strs = parse_structures_from_poscars(poscars)
    assert len(strs) == 2
    st = strs[0]
    assert st.positions.shape == (3, 8)
    assert st.types == [0, 0, 0, 0, 1, 1, 1, 1]
    assert st.elements == ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]


# TODO: vasprun from MD, OUTCAR, DOSCAR
