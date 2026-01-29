"""Tests of dataset_utils."""

from pathlib import Path

import numpy as np

from pypolymlp.core.dataset_utils import DatasetDFT, permute_atoms

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


def test_dataset_dft(structure_rocksalt):
    """Test for DatasetDFT class."""
    st = structure_rocksalt
    data = DatasetDFT()
    data.structures = [st, st, st]
    data.energies = np.array([0.1, 0.2, 0.3])
    data.forces = np.ones(144)
    data.stresses = np.random.random(18)
    data.volumes = np.random.random(3) * 10
    data.total_n_atoms = np.array([8, 24, 16])
    data.elements = ["Mg", "O"]
    data.files = ["f1", "f2", "f3"]

    data2 = data.slice(0, 2)
    np.testing.assert_allclose(data2.energies, [0.1, 0.2])
    np.testing.assert_allclose(data2.total_n_atoms, [8, 24])
    np.testing.assert_allclose(data2.forces, data.forces[:96])
    np.testing.assert_allclose(data2.stresses, data.stresses[:12])
    np.testing.assert_allclose(data2.volumes, data.volumes[:2])

    data.apply_atomic_energy([0.01, 0.02])
    np.testing.assert_allclose(data.energies, [-0.02, 0.08, 0.18])

    train, test = data.split(train_ratio=0.5)
    np.testing.assert_allclose(train.energies, [0.08, 0.18])
    np.testing.assert_allclose(train.total_n_atoms, [24, 16])
    np.testing.assert_allclose(train.forces, data.forces[24:])
    np.testing.assert_allclose(train.stresses, data.stresses[6:])
    np.testing.assert_allclose(train.volumes, data.volumes[1:])
    assert len(train.structures) == 2

    np.testing.assert_allclose(test.energies, [-0.02])
    np.testing.assert_allclose(test.total_n_atoms, [8])
    np.testing.assert_allclose(test.forces, data.forces[:24])
    np.testing.assert_allclose(test.stresses, data.stresses[:6])
    np.testing.assert_allclose(test.volumes, data.volumes[:1])
    assert len(test.structures) == 1

    data.sort()
    np.testing.assert_equal(data.total_n_atoms, [8, 16, 24])
    np.testing.assert_allclose(data.energies, [-0.02, 0.18, 0.08])
    assert data.forces.shape[0] == 144
    assert data.stresses.shape[0] == 18
    assert data.volumes.shape[0] == 3
    assert len(data.structures) == 3


def test_dataset_dft_initialize(structure_rocksalt):
    """Test for DatasetDFT class."""
    st = structure_rocksalt
    structures = [st, st, st]
    energies = np.array([0.1, 0.2, 0.3])
    forces = [np.ones((3, 8)), np.ones((3, 8)), np.ones((3, 8))]
    stresses = np.random.random((3, 3, 3))
    elements = ["Mg", "O"]

    data = DatasetDFT(
        structures,
        energies,
        forces=forces,
        stresses=stresses,
        element_order=elements,
    )
    np.testing.assert_allclose(data.energies, energies)
    np.testing.assert_equal(data.elements, elements)
    assert data.forces.shape[0] == 72
    assert data.stresses.shape[0] == 18
    assert data.volumes.shape[0] == 3
    assert data.total_n_atoms.shape[0] == 3
    assert len(data.files) == 3
    assert data.exist_force == True
    assert data.exist_stress == True
