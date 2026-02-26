"""Tests of functions to get elemental structures."""

import numpy as np

from pypolymlp.calculator.auto.autocalc_utils import Prototype
from pypolymlp.calculator.auto.structures_element import (
    get_structure_list_element,
    get_structure_type_element,
    set_structure,
)


def test_set_structure():
    """Test set_structure."""
    axis = np.eye(3) * 3.0
    positions = np.random.random((3, 6))
    element_strings = ["Ag"]
    st = set_structure(axis, positions, element_strings)
    np.testing.assert_allclose(st.axis, axis)
    np.testing.assert_allclose(st.positions, positions)
    np.testing.assert_equal(st.n_atoms, [6])
    np.testing.assert_equal(st.types, [0, 0, 0, 0, 0, 0])
    np.testing.assert_equal(st.elements, ["Ag"] * 6)


def test_get_structure_list_element():
    """Test get_structure_list_element."""
    elements = ["Ag"]
    structure_list = get_structure_list_element(elements)
    for st in structure_list:
        assert isinstance(st, Prototype)
    assert len(structure_list) == 18


def test_get_structure_type_element():
    """Test get_structure_type_element."""
    structure_type = get_structure_type_element()
    assert structure_type[105489] == "FeB"
