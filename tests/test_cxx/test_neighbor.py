"""Tests of Neighbor C++ class."""

import numpy as np
import pytest

from pypolymlp.cxx_wrapper.api_neighbor import (
    Neighbor,
    NeighborCell,
    NeighborFull,
    NeighborHalf,
)


def test_compute_neighbor_list(structure_rocksalt):
    """Test for neighbor distance list."""
    neigh = Neighbor(structure_rocksalt, cutoff=6.0)
    distances = neigh.distances
    assert len(distances) == 8
    assert len(distances[0][0]) == 54
    assert len(distances[0][1]) == 38
    assert np.count_nonzero(np.isclose(distances[0][0], 2.8284271247461903)) == 12
    assert np.count_nonzero(np.isclose(distances[0][0], 4.0)) == 6
    assert np.count_nonzero(np.isclose(distances[0][0], 4.898979485566356)) == 24
    assert np.count_nonzero(np.isclose(distances[0][0], 5.656854249492381)) == 12
    assert np.count_nonzero(np.isclose(distances[0][1], 2.0)) == 6
    assert np.count_nonzero(np.isclose(distances[0][1], 3.46410162)) == 8
    assert np.count_nonzero(np.isclose(distances[0][1], 4.47213595)) == 24

    differences = neigh.differences
    assert len(differences) == 8
    assert len(differences[0]) == 2
    assert len(differences[0][0]) == 54
    assert len(differences[0][1]) == 38
    assert np.sum(np.square(differences[0][0])) == pytest.approx(1152)
    assert np.sum(np.square(differences[0][1])) == pytest.approx(600)
    assert np.sum(np.square(differences[4][0])) == pytest.approx(600)
    assert np.sum(np.square(differences[4][1])) == pytest.approx(1152)

    neighbor_atoms = neigh.neighbor_atoms
    assert len(neighbor_atoms) == 8
    assert len(neighbor_atoms[0]) == 2
    assert len(neighbor_atoms[0][0]) == 54
    assert len(neighbor_atoms[0][1]) == 38
    assert np.sum(neighbor_atoms[0][0]) == 72
    assert np.sum(neighbor_atoms[0][1]) == 206
    assert np.sum(neighbor_atoms[1][0]) == 78
    assert np.sum(neighbor_atoms[1][1]) == 208
    assert np.sum(neighbor_atoms[2][0]) == 84
    assert np.sum(neighbor_atoms[2][1]) == 210
    assert np.sum(neighbor_atoms[3][0]) == 90
    assert np.sum(neighbor_atoms[3][1]) == 212
    assert np.sum(neighbor_atoms[4][0]) == 54
    assert np.sum(neighbor_atoms[4][1]) == 288
    assert np.sum(neighbor_atoms[5][0]) == 56
    assert np.sum(neighbor_atoms[5][1]) == 294
    assert np.sum(neighbor_atoms[6][0]) == 58
    assert np.sum(neighbor_atoms[6][1]) == 300
    assert np.sum(neighbor_atoms[7][0]) == 60
    assert np.sum(neighbor_atoms[7][1]) == 306


def test_neighbor_half_list(structure_rocksalt):
    """Test for neighbor half list."""
    neigh = NeighborHalf(structure_rocksalt, cutoff=6.0, use_openmp=True)

    differences = neigh.differences
    assert len(differences) == 8
    assert np.array(differences[0]).shape == (9, 3)
    assert np.array(differences[1]).shape == (21, 3)
    assert np.array(differences[2]).shape == (33, 3)
    assert np.array(differences[3]).shape == (45, 3)
    assert np.array(differences[4]).shape == (47, 3)
    assert np.array(differences[5]).shape == (59, 3)
    assert np.array(differences[6]).shape == (71, 3)
    assert np.array(differences[7]).shape == (83, 3)

    assert differences[2][5][0] == pytest.approx(-2.0)
    assert differences[2][5][1] == pytest.approx(4.0)
    assert differences[2][5][2] == pytest.approx(2.0)

    neighbor_atoms = neigh.neighbor_atoms
    assert len(neighbor_atoms) == 8
    assert len(neighbor_atoms[0]) == 9
    assert len(neighbor_atoms[1]) == 21
    assert len(neighbor_atoms[2]) == 33
    assert len(neighbor_atoms[3]) == 45
    assert len(neighbor_atoms[4]) == 47
    assert len(neighbor_atoms[5]) == 59
    assert len(neighbor_atoms[6]) == 71
    assert len(neighbor_atoms[7]) == 83

    assert np.sum(neighbor_atoms[0]) == 0
    assert np.sum(neighbor_atoms[1]) == 9
    assert np.sum(neighbor_atoms[2]) == 30
    assert np.sum(neighbor_atoms[3]) == 63
    assert np.sum(neighbor_atoms[4]) == 90
    assert np.sum(neighbor_atoms[5]) == 149
    assert np.sum(neighbor_atoms[6]) == 220
    assert np.sum(neighbor_atoms[7]) == 303


def test_neighbor_half_list_2(structure_rocksalt):
    """Test for neighbor half list."""
    neigh = NeighborHalf(structure_rocksalt, cutoff=6.0, use_openmp=False)

    differences = neigh.differences
    assert len(differences) == 8
    assert np.array(differences[0]).shape == (9, 3)
    assert np.array(differences[1]).shape == (21, 3)
    assert np.array(differences[2]).shape == (33, 3)
    assert np.array(differences[3]).shape == (45, 3)
    assert np.array(differences[4]).shape == (47, 3)
    assert np.array(differences[5]).shape == (59, 3)
    assert np.array(differences[6]).shape == (71, 3)
    assert np.array(differences[7]).shape == (83, 3)

    assert differences[2][5][0] == pytest.approx(-2.0)
    assert differences[2][5][1] == pytest.approx(4.0)
    assert differences[2][5][2] == pytest.approx(2.0)

    neighbor_atoms = neigh.neighbor_atoms
    assert len(neighbor_atoms) == 8
    assert len(neighbor_atoms[0]) == 9
    assert len(neighbor_atoms[1]) == 21
    assert len(neighbor_atoms[2]) == 33
    assert len(neighbor_atoms[3]) == 45
    assert len(neighbor_atoms[4]) == 47
    assert len(neighbor_atoms[5]) == 59
    assert len(neighbor_atoms[6]) == 71
    assert len(neighbor_atoms[7]) == 83

    assert np.sum(neighbor_atoms[0]) == 0
    assert np.sum(neighbor_atoms[1]) == 9
    assert np.sum(neighbor_atoms[2]) == 30
    assert np.sum(neighbor_atoms[3]) == 63
    assert np.sum(neighbor_atoms[4]) == 90
    assert np.sum(neighbor_atoms[5]) == 149
    assert np.sum(neighbor_atoms[6]) == 220
    assert np.sum(neighbor_atoms[7]) == 303


def test_compute_neighbor_full_list(structure_rocksalt):
    """Test for neighbor distance list."""
    neigh = NeighborFull(structure_rocksalt, cutoff=6.0)
    distances = neigh.distances
    assert len(distances) == 8
    assert len(distances[0][0]) == 54
    assert len(distances[0][1]) == 38
    assert np.count_nonzero(np.isclose(distances[0][0], 2.8284271247461903)) == 12
    assert np.count_nonzero(np.isclose(distances[0][0], 4.0)) == 6
    assert np.count_nonzero(np.isclose(distances[0][0], 4.898979485566356)) == 24
    assert np.count_nonzero(np.isclose(distances[0][0], 5.656854249492381)) == 12
    assert np.count_nonzero(np.isclose(distances[0][1], 2.0)) == 6
    assert np.count_nonzero(np.isclose(distances[0][1], 3.46410162)) == 8
    assert np.count_nonzero(np.isclose(distances[0][1], 4.47213595)) == 24

    differences = neigh.differences
    assert len(differences) == 8
    assert len(differences[0]) == 2
    assert len(differences[0][0]) == 54
    assert len(differences[0][1]) == 38
    assert np.sum(np.square(differences[0][0])) == pytest.approx(1152)
    assert np.sum(np.square(differences[0][1])) == pytest.approx(600)
    assert np.sum(np.square(differences[4][0])) == pytest.approx(600)
    assert np.sum(np.square(differences[4][1])) == pytest.approx(1152)

    neighbor_atoms = neigh.neighbor_atoms
    assert len(neighbor_atoms) == 8
    assert len(neighbor_atoms[0]) == 2
    assert len(neighbor_atoms[0][0]) == 54
    assert len(neighbor_atoms[0][1]) == 38
    assert np.sum(neighbor_atoms[0][0]) == 72
    assert np.sum(neighbor_atoms[0][1]) == 206
    assert np.sum(neighbor_atoms[1][0]) == 78
    assert np.sum(neighbor_atoms[1][1]) == 208
    assert np.sum(neighbor_atoms[2][0]) == 84
    assert np.sum(neighbor_atoms[2][1]) == 210
    assert np.sum(neighbor_atoms[3][0]) == 90
    assert np.sum(neighbor_atoms[3][1]) == 212
    assert np.sum(neighbor_atoms[4][0]) == 54
    assert np.sum(neighbor_atoms[4][1]) == 288
    assert np.sum(neighbor_atoms[5][0]) == 56
    assert np.sum(neighbor_atoms[5][1]) == 294
    assert np.sum(neighbor_atoms[6][0]) == 58
    assert np.sum(neighbor_atoms[6][1]) == 300
    assert np.sum(neighbor_atoms[7][0]) == 60
    assert np.sum(neighbor_atoms[7][1]) == 306


def test_neighbor_cell(structure_rocksalt):
    """Test for neighbor distance list."""
    neigh = NeighborCell(structure_rocksalt, cutoff=6.0)
    np.testing.assert_allclose(neigh.axis, structure_rocksalt.axis)

    cartesian = structure_rocksalt.axis @ structure_rocksalt.positions
    np.testing.assert_allclose(neigh.positions_cartesian, cartesian)

    trans = neigh.translations
    assert len(trans) == 147

    neigh = NeighborCell(structure_rocksalt, cutoff=8.0)
    trans = neigh.translations
    assert len(trans) == 203

    neigh = NeighborCell(structure_rocksalt, cutoff=16.0)
    trans = neigh.translations
    assert len(trans) == 751
