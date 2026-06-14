"""Tests of Neighbor C++ class."""

import numpy as np

from pypolymlp.cxx.api_neighbor import Neighbor


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
