"""Tests of dataset_utils."""

from pathlib import Path

import numpy as np

from pypolymlp.core.dataset_utils import DatasetDFT

cwd = Path(__file__).parent


def test_dataset_dft(structure_rocksalt):
    """Test for DatasetDFT class."""
    energies = np.array([0.1, 0.2, 0.3])
    total_n_atoms = np.array([8, 24, 16])
    forces = np.ones(144)
    stresses = np.random.random(18)
    volumes = np.random.random(3) * 10

    st = structure_rocksalt
    structures = [st, st, st]
    elements = ["Mg", "O"]

    data = DatasetDFT(
        energies,
        forces,
        stresses,
        volumes,
        structures,
        total_n_atoms,
        elements,
    )
    data2 = data.slice(0, 2)
    np.testing.assert_allclose(data2.energies, [0.1, 0.2])
    np.testing.assert_allclose(data2.total_n_atoms, [8, 24])
    np.testing.assert_allclose(data2.forces, forces[:96])
    np.testing.assert_allclose(data2.stresses, stresses[:12])
    np.testing.assert_allclose(data2.volumes, volumes[:2])

    data.apply_atomic_energy([0.01, 0.02])
    np.testing.assert_allclose(data.energies, [-0.02, 0.08, 0.18])

    train, test = data.split(train_ratio=0.5)
    np.testing.assert_allclose(train.energies, [0.08, 0.18])
    np.testing.assert_allclose(train.total_n_atoms, [24, 16])
    np.testing.assert_allclose(train.forces, forces[24:])
    np.testing.assert_allclose(train.stresses, stresses[6:])
    np.testing.assert_allclose(train.volumes, volumes[1:])
    assert len(train.structures) == 2

    np.testing.assert_allclose(test.energies, [-0.02])
    np.testing.assert_allclose(test.total_n_atoms, [8])
    np.testing.assert_allclose(test.forces, forces[:24])
    np.testing.assert_allclose(test.stresses, stresses[:6])
    np.testing.assert_allclose(test.volumes, volumes[:1])
    assert len(test.structures) == 1

    data.sort()
    np.testing.assert_equal(data.total_n_atoms, [8, 16, 24])
    np.testing.assert_allclose(data.energies, [-0.02, 0.18, 0.08])
    assert data.forces.shape[0] == 144
    assert data.stresses.shape[0] == 18
    assert data.volumes.shape[0] == 3
    assert len(data.structures) == 3
