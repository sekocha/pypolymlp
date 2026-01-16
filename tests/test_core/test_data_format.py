"""Tests of data_format."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.data_format import PolymlpStructure

cwd = Path(__file__).parent


def test_polymlp_structure(structure_rocksalt):
    """Test for PolymlpStructure class."""
    st = structure_rocksalt
    assert st.axis.shape == (3, 3)
    assert st.positions.shape == (3, 8)
    np.testing.assert_equal(st.n_atoms, [4, 4])
    np.testing.assert_equal(st.types, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_equal(st.elements, ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"])
    assert st.volume == pytest.approx(64.0)

    _ = PolymlpStructure(
        axis=st.axis,
        positions=st.positions,
        n_atoms=st.n_atoms,
        elements=st.elements,
        types=st.types,
        volume=64.0,
        supercell_matrix=np.eye(3),
        n_unitcells=1,
        comment="test",
        name="POSCAR-rocksalt",
        masses=[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    )

    st_rev = copy.deepcopy(st)
    st_rev.types = [0, 0, 1, 1, 0, 0, 1, 1]
    st_rev.elements = ["Mg", "Mg", "O", "O", "Mg", "Mg", "O", "O"]
    st_rev.positions = st_rev.positions[:, np.array([0, 1, 4, 5, 2, 3, 6, 7])]
    st_rev = st_rev.reorder()

    np.testing.assert_allclose(st.axis, st_rev.axis)
    np.testing.assert_allclose(st.positions, st_rev.positions)
    np.testing.assert_equal(st.n_atoms, st_rev.n_atoms)
    np.testing.assert_equal(st.types, st_rev.types)
    np.testing.assert_equal(st.elements, st_rev.elements)
