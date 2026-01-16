"""Tests of structure utility functions."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.structure_utils import (
    calc_positions_cartesian,
    get_lattice_constants,
    get_reciprocal_axis,
    introduce_disp,
    isotropic_volume_change,
    multiple_isotropic_volume_changes,
    multiple_random_deformation,
    random_deformation,
    refine_positions,
    remove,
    remove_atom,
    reset_types,
    sort_wrt_types,
    supercell,
    supercell_diagonal,
    swap_elements,
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


def test_disp(structure_rocksalt):
    """Test for displacements."""
    st = copy.deepcopy(structure_rocksalt)
    st = introduce_disp(st)
    diff = st.positions - structure_rocksalt.positions
    np.testing.assert_equal(np.abs(diff) < 2e-3, True)


def test_volumes(structure_rocksalt):
    """Test for volume changes."""
    st_vol = isotropic_volume_change(structure_rocksalt, eps=1.1)
    assert st_vol.volume == pytest.approx(70.4)
    st_vols = multiple_isotropic_volume_changes(structure_rocksalt, n_eps=2)
    assert len(st_vols) == 2
    assert st_vols[0].volume == pytest.approx(44.8)
    assert st_vols[1].volume == pytest.approx(128.0)


def test_supercell(structure_rocksalt):
    """Test for supercell functions."""
    sup_mat = [[1, 0, 0], [0, 1, 0], [1, 0, 2]]
    sup = supercell(structure_rocksalt, supercell_matrix=sup_mat)
    np.testing.assert_equal(sup.n_atoms, [8, 8])

    sup = supercell_diagonal(structure_rocksalt, size=(1, 1, 2))
    np.testing.assert_equal(sup.n_atoms, [8, 8])


def test_remove_element(structure_rocksalt):
    """Test for element removal."""
    st = copy.deepcopy(structure_rocksalt)
    st_rem = remove(st, 0)
    assert st_rem.positions.shape == (3, 4)
    np.testing.assert_equal(st_rem.n_atoms, [4])
    np.testing.assert_equal(st_rem.types, 1)
    np.testing.assert_equal(st_rem.elements, "O")


def test_remove_atom(structure_rocksalt):
    """Test for atom removal."""
    st = copy.deepcopy(structure_rocksalt)
    st_rem = remove_atom(st, 3)
    assert st_rem.positions.shape == (3, 7)
    np.testing.assert_equal(st_rem.n_atoms, [3, 4])
    np.testing.assert_equal(st_rem.types, [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_equal(st_rem.elements, ["Mg"] * 3 + ["O"] * 4)


def test_swap_elements(structure_rocksalt):
    """Test for swap positions and atoms."""
    st = copy.deepcopy(structure_rocksalt)
    st = swap_elements(st, order=[1, 0])
    order = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    np.testing.assert_allclose(st.positions, structure_rocksalt.positions[:, order])
    np.testing.assert_equal(st.n_atoms, structure_rocksalt.n_atoms)
    np.testing.assert_equal(st.types, structure_rocksalt.types)
    np.testing.assert_equal(st.elements, structure_rocksalt.elements)


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


def test_random_deformation(structure_rocksalt):
    """Test for random_deformation."""
    st = copy.deepcopy(structure_rocksalt)
    st_list = [st, st]
    _ = random_deformation(st, max_deform=0.01)
    _ = multiple_random_deformation(st_list, max_deform=0.01)


def test_sort_wrt_type(structure_rocksalt):
    """Test for sort_wrt_types."""
    st = copy.deepcopy(structure_rocksalt)
    st.types = [1, 1, 1, 1, 0, 0, 0, 0]
    st, order = sort_wrt_types(st, return_ids=True)
    np.testing.assert_allclose(st.positions, structure_rocksalt.positions[:, order])
    np.testing.assert_equal(st.n_atoms, structure_rocksalt.n_atoms)
    np.testing.assert_equal(st.types, [0] * 4 + [1] * 4)
    np.testing.assert_equal(st.elements, ["O"] * 4 + ["Mg"] * 4)
