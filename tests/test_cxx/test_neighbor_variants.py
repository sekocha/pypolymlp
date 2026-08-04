"""Tests of Neighbor C++ class for structure variants."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.cxx.wrapper.api_neighbor import NeighborCell, NeighborFull

cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"

str1 = Poscar(path_files + "POSCAR-BiGd2").structure


def test_compute_neighbor_BiGd():
    """Test for neighbor distance list."""
    neigh = NeighborFull(str1, cutoff=6.0)
    distances = neigh.distances
    assert len(distances) == 30
    assert len(distances[0]) == 2
    assert len(distances[0][0]) == 8
    assert len(distances[15][1]) == 17
    assert np.sum(distances[0][0]) == pytest.approx(38.163579855514044)
    assert np.sum(distances[0][1]) == pytest.approx(83.65951693527006)

    differences = neigh.differences
    assert len(differences) == 30
    assert len(differences[0]) == 2
    assert len(differences[0][0]) == 8
    assert len(differences[15][1]) == 17
    assert np.sum(np.square(differences[0][0])) == pytest.approx(187.52873463482072)
    assert np.sum(np.square(differences[0][1])) == pytest.approx(415.3567695554637)
    assert np.sum(np.square(differences[15][0])) == pytest.approx(207.80873466782282)
    assert np.sum(np.square(differences[15][1])) == pytest.approx(381.07889949085563)

    neighbor_atoms = neigh.neighbor_atoms
    assert len(neighbor_atoms) == 30
    assert len(neighbor_atoms[0]) == 2
    assert len(neighbor_atoms[0][0]) == 8
    assert len(neighbor_atoms[15][1]) == 17
    assert np.sum(neighbor_atoms[0][0]) == 15
    assert np.sum(neighbor_atoms[0][1]) == 325
    assert np.sum(neighbor_atoms[15][0]) == 32
    assert np.sum(neighbor_atoms[15][1]) == 302


def test_neighbor_cell_BiGd():
    """Test for neighbor cell list."""
    neigh = NeighborCell(str1, cutoff=6.0)
    np.testing.assert_allclose(neigh.axis, str1.axis)

    cartesian = str1.axis @ str1.positions
    np.testing.assert_allclose(neigh.positions_cartesian, cartesian)

    trans = neigh.translations
    assert len(trans) == 91

    neigh = NeighborCell(str1, cutoff=8.0)
    trans = neigh.translations
    assert len(trans) == 117

    neigh = NeighborCell(str1, cutoff=16.0)
    trans = neigh.translations
    assert len(trans) == 281
