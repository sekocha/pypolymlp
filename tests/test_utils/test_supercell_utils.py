"""Tests of supercell utility functions."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.supercell_utils import (
    _is_diagonal,
    get_supercell,
    get_supercell_size,
)

cwd = Path(__file__).parent


def test_is_diagonal():
    """Test _is_diagonal."""
    mat = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    assert not _is_diagonal(mat)
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    assert _is_diagonal(mat)


@pytest.mark.filterwarnings("ignore:.*symmetries of supercell.*")
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
    np.testing.assert_allclose(sup.positions.T, positions_true)

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
    np.testing.assert_allclose(sup.positions.T, positions_true)

    sup = get_supercell(structure_rocksalt, (1, 1, 2), use_phonopy=True)
    np.testing.assert_equal(sup.n_atoms, [8, 8])
    np.testing.assert_allclose(sup.positions.T, positions_true)

    sup = get_supercell(structure_rocksalt, [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    np.testing.assert_equal(sup.n_atoms, [8, 8])
    np.testing.assert_allclose(sup.positions.T, positions_true)


@pytest.mark.filterwarnings("ignore:.*symmetries of supercell.*")
def test_supercell_algorithms(structure_rocksalt):
    """Test for equality of algorithms in pypolymlp and phonopy."""
    sup1 = get_supercell(structure_rocksalt, (2, 2, 3))
    sup2 = get_supercell(structure_rocksalt, (2, 2, 3), use_phonopy=True)

    for p1, p2 in zip(sup1.positions.T, sup2.positions.T):
        print(p1, p2)
    np.testing.assert_allclose(sup1.positions, sup2.positions, atol=1e-7)


def test_get_supercell_size():
    """Test get_supercell_size."""
    assert get_supercell_size([2, 3, 4]) == 24
    assert get_supercell_size(np.eye(3) * 2) == 8
