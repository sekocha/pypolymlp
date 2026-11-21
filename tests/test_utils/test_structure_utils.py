"""Tests of structure utility functions."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.structure_utils import (
    introduce_disp,
    isotropic_volume_change,
    multiple_isotropic_volume_changes,
    multiple_random_deformation,
    random_deformation,
    remove,
    remove_atom,
    supercell,
    supercell_diagonal,
    swap_elements,
)

cwd = Path(__file__).parent


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


@pytest.mark.filterwarnings("ignore:.*symmetries of supercell.*")
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


def test_random_deformation(structure_rocksalt):
    """Test for random_deformation."""
    st = copy.deepcopy(structure_rocksalt)
    st_list = [st, st]
    _ = random_deformation(st, max_deform=0.01)
    _ = multiple_random_deformation(st_list, max_deform=0.01)
