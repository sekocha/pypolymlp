"""Tests of functions to get structure types."""

from pypolymlp.calculator.auto.structures_types import (
    get_structure_types,
    get_structure_types_binary,
    get_structure_types_element,
)


def test_get_structure_type():
    """Test get_structure_type_element."""
    structure_type = get_structure_types()
    assert structure_type["105489"] == "FeB"
    assert structure_type["100654-01"] == "BiSe"


def test_get_structure_type_element():
    """Test get_structure_type_element."""
    structure_type = get_structure_types_element()
    assert structure_type["105489"] == "FeB"


def test_get_structure_type_binary():
    """Test get_structure_type_binary."""
    structure_type = get_structure_types_binary()
    assert structure_type["100654-01"] == "BiSe"
