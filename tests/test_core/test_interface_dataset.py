"""Tests of interface_dataset."""

from pathlib import Path

import numpy as np

from pypolymlp.core.interface_datasets import permute_atoms, set_dataset_from_structures

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


def test_set_dataset_from_structures(structure_rocksalt):
    """Test for set_dataset_from_structures."""
    energies = [2.0, 1.0]
    forces = [
        np.random.random(structure_rocksalt.positions.shape),
        np.random.random(structure_rocksalt.positions.shape),
    ]
    structures = [structure_rocksalt, structure_rocksalt]

    element_order = ("Mg", "O")
    dft1 = set_dataset_from_structures(
        structures,
        energies,
        forces,
        element_order=element_order,
    )
    forces1 = dft1.forces.reshape((2, 8, 3))
    positions1 = [st.positions for st in dft1.structures]

    element_order = ("O", "Mg")
    dft2 = set_dataset_from_structures(
        structures,
        energies,
        forces,
        element_order=element_order,
    )
    order = [4, 5, 6, 7, 0, 1, 2, 3]
    forces2 = dft2.forces.reshape((2, 8, 3))[:, order, :]
    positions2 = [st.positions[:, order] for st in dft2.structures]

    np.testing.assert_allclose(forces1, forces2)
    np.testing.assert_allclose(positions1, positions2)
