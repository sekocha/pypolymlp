"""Tests of structure utility functions."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.utils.structure_core_utils import (
    calc_positions_cartesian,
    get_lattice_constants,
    get_reciprocal_axis,
    refine_positions,
    replace_types,
    reset_types,
    sort_wrt_types,
    triangulation_axis,
)

cwd = Path(__file__).parent


def test_refine_positions(structure_rocksalt):
    """Test for refine_positions."""
    pos = structure_rocksalt.positions
    pos2 = pos + 2.0
    pos2 = refine_positions(positions=pos2)
    np.testing.assert_allclose(pos2, pos, atol=1e-6)


def test_functions(structure_rocksalt):
    """Test for reset_types and others."""
    structure_rocksalt = reset_types(structure_rocksalt)

    pos_c = calc_positions_cartesian(structure_rocksalt)
    np.testing.assert_allclose(pos_c, structure_rocksalt.positions * 4.0, atol=1e-6)

    rec = get_reciprocal_axis(structure_rocksalt)
    np.testing.assert_allclose(rec, np.eye(3) * 1.57079633, atol=1e-6)


def test_lattice_consts(structure_rocksalt):
    """Test for get_lattice_constants."""
    consts = get_lattice_constants(structure_rocksalt)
    np.testing.assert_allclose(consts, [4.0, 4.0, 4.0, 0.0, 0.0, 0.0])


def test_triangulation_axis(structure_rocksalt):
    """Test for triangulation_axis."""
    st = copy.deepcopy(structure_rocksalt)
    st.axis[0][1] = st.axis[1][0] = 0.02
    st.axis[0][2] = st.axis[2][0] = 0.03
    st.axis[1][2] = st.axis[2][1] = 0.003
    st = triangulation_axis(st)
    np.testing.assert_allclose(
        st.axis,
        [
            [4.0001625, 0.04002087, 0.06001256],
            [0.0, 3.99985091, 0.00554977],
            [0.0, 0.0, 3.99965957],
        ],
        atol=1e-6,
    )


def test_sort_wrt_type(structure_rocksalt):
    """Test for sort_wrt_types."""
    st = copy.deepcopy(structure_rocksalt)
    st.types = [1, 1, 1, 1, 0, 0, 0, 0]
    st, order = sort_wrt_types(st, return_ids=True)
    np.testing.assert_allclose(st.positions, structure_rocksalt.positions[:, order])
    np.testing.assert_equal(st.n_atoms, structure_rocksalt.n_atoms)
    np.testing.assert_equal(st.types, [0] * 4 + [1] * 4)
    np.testing.assert_equal(st.elements, ["O"] * 4 + ["Mg"] * 4)


def test_replace_types(structure_rocksalt):
    """Test for replace_types."""
    structure_rocksalt_copy = copy.deepcopy(structure_rocksalt)
    elements = ("O", "Mg")
    st = replace_types(structure_rocksalt_copy, elements)
    np.testing.assert_equal(st.types, [1, 1, 1, 1, 0, 0, 0, 0])


def test_replace_types_spins(structure_rocksalt):
    """Test for replace_types with spin configurations."""
    structure_rocksalt_copy = copy.deepcopy(structure_rocksalt)
    structure_rocksalt_copy.types = [0, 1, 1, 0, 2, 2, 2, 2]
    elements = ("O", "Mg", "Mg")
    st = replace_types(structure_rocksalt_copy, elements)
    np.testing.assert_equal(st.types, [1, 2, 2, 1, 0, 0, 0, 0])
