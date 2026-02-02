"""Tests of supercell utility functions."""

from pathlib import Path

import numpy as np

from pypolymlp.utils.supercell_utils import _is_diagonal, get_supercell

cwd = Path(__file__).parent


def test_is_diagonal():
    """Test _is_diagonal."""
    mat = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    assert not _is_diagonal(mat)
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    assert _is_diagonal(mat)


def test_supercell(structure_rocksalt):
    """Test for supercell functions."""
    sup_mat = [[1, 0, 0], [0, 1, 0], [1, 0, 2]]
    sup = get_supercell(structure_rocksalt, sup_mat)
    np.testing.assert_equal(sup.n_atoms, [8, 8])
    positions_true = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.75],
        [0.5, 0.5, 0.25],
        [0.0, 0.0, 0.25],
        [0.0, 0.0, 0.75],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.75],
        [0.5, 0.0, 0.25],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
    ]
    np.testing.assert_equal(sup.positions.T, positions_true)

    sup = get_supercell(structure_rocksalt, (1, 1, 2))
    np.testing.assert_equal(sup.n_atoms, [8, 8])
    positions_true = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
        [0.5, 0.0, 0.25],
        [0.5, 0.0, 0.75],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.25],
        [0.0, 0.0, 0.75],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.25],
        [0.5, 0.5, 0.75],
    ]
    np.testing.assert_equal(sup.positions.T, positions_true)

    sup = get_supercell(structure_rocksalt, (1, 1, 2), use_phonopy=True)
    np.testing.assert_equal(sup.n_atoms, [8, 8])
    np.testing.assert_equal(sup.positions.T, positions_true)
