"""Tests of functions to get binary structures."""

import numpy as np

from pypolymlp.calculator.auto.autocalc_utils import Prototype
from pypolymlp.calculator.auto.structures_binary import (
    get_structure_list_binary,
    get_structure_type_binary,
    set_structure,
)


def test_set_structure():
    """Test set_structure."""
    axis = np.eye(3) * 3.0
    positions = np.random.random((3, 6))
    n_atoms = [2, 4]
    element_strings = ("Ag", "Au")
    st = set_structure(axis, positions, n_atoms, element_strings)
    np.testing.assert_allclose(st.axis, axis)
    np.testing.assert_allclose(st.positions, positions)
    np.testing.assert_equal(st.n_atoms, n_atoms)
    np.testing.assert_equal(st.types, [0, 0, 1, 1, 1, 1])
    np.testing.assert_equal(st.elements, ["Ag", "Ag", "Au", "Au", "Au", "Au"])


def test_get_structure_list_binary():
    """Test get_structure_list_binary."""
    elements = ("Ag", "Au")
    structure_list = get_structure_list_binary(elements)
    for st in structure_list:
        assert isinstance(st, Prototype)
    assert len(structure_list) == 33


def test_get_structure_type_binary():
    """Test get_structure_type_binary."""
    structure_type = get_structure_type_binary()
    assert structure_type["100654-01"] == "BiSe"
