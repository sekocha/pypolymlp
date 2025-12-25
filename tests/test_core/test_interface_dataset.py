"""Tests of interface_dataset."""

from pathlib import Path

import numpy as np

from pypolymlp.core.interface_datasets import permute_atoms

cwd = Path(__file__).parent


def test_permute_atoms(structure_rocksalt):
    """Test for permute_atoms."""
    force = np.random.random(structure_rocksalt.positions.shape)
    element_order = ("O", "Mg")
    st, force_permute = permute_atoms(structure_rocksalt, force, element_order)

    order = np.array([4, 5, 6, 7, 0, 1, 2, 3])
    np.testing.assert_allclose(force_permute, force[:, order])
    np.testing.assert_allclose(st.positions, structure_rocksalt.positions[:, order])
    np.testing.assert_equal(st.n_atoms, structure_rocksalt.n_atoms)
    np.testing.assert_equal(st.types, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_equal(st.elements, np.array(structure_rocksalt.elements)[order])
