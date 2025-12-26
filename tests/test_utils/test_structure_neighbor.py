"""Tests of functions for neighbor attributes."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.structure_neighbor import (
    compute_neighbor_list,
    find_active_distances,
    find_minimum_distance,
    get_coordination_numbers,
)

cwd = Path(__file__).parent


def test_find_active_distances(structure_rocksalt):
    """Test for find_active_distances."""
    distance = find_active_distances(structure_rocksalt, cutoff=6.0, decimals=1)
    np.testing.assert_allclose(distance[("Mg", "Mg")], [2.8, 4.0, 4.9, 5.7])
    np.testing.assert_allclose(distance[("Mg", "O")], [2.0, 3.5, 4.5])
    np.testing.assert_allclose(distance[("O", "O")], [2.8, 4.0, 4.9, 5.7])


def test_compute_neighbor_list(structure_rocksalt):
    """Test for neighbor distance list."""
    distances = compute_neighbor_list(structure_rocksalt, cutoff=6.0)
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


def test_find_minimum_distance(structure_rocksalt):
    """Test for find_minimum_distance."""
    min1 = find_minimum_distance(structure_rocksalt, each_atom=False)
    assert min1 == pytest.approx(2.0)
    min1 = find_minimum_distance(structure_rocksalt, each_atom=True)
    np.testing.assert_allclose(
        min1,
        [
            2.828427,
            2.0,
            2.828427,
            2.0,
            2.828427,
            2.0,
            2.828427,
            2.0,
            2.0,
            2.828427,
            2.0,
            2.828427,
            2.0,
            2.828427,
            2.0,
            2.828427,
        ],
        atol=1e-6,
    )


def test_get_coordination_numbers(structure_rocksalt):
    """Test for get_coordination_numbers."""
    distances = compute_neighbor_list(structure_rocksalt, cutoff=6.0)
    coord = get_coordination_numbers(distances)
    assert coord == [54, 38, 54, 38, 54, 38, 54, 38, 38, 54, 38, 54, 38, 54, 38, 54]
