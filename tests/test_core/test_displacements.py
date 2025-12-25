"""Tests of convert_disps_to_positions."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.core.displacements import (
    convert_disps_to_positions,
    convert_positions_to_disps,
    generate_random_const_displacements,
    get_structures_from_displacements,
)

cwd = Path(__file__).parent


def test_convert(structure_rocksalt):
    """Test for convert_disps_to_positions and convert_positions_to_disps."""
    st = structure_rocksalt
    shape = (2, 3, st.positions.shape[1])
    disps = np.ones(shape) * 0.02
    positions_new = convert_disps_to_positions(disps, st.axis, st.positions)
    np.testing.assert_allclose(
        positions_new[0][0], [0.005, 0.005, 0.505, 0.505, 0.005, 0.005, 0.505, 0.505]
    )
    np.testing.assert_allclose(
        positions_new[0][1], [0.005, 0.505, 0.005, 0.505, 0.005, 0.505, 0.005, 0.505]
    )
    np.testing.assert_allclose(
        positions_new[0][2], [0.005, 0.505, 0.505, 0.005, 0.505, 0.005, 0.005, 0.505]
    )

    st2 = copy.deepcopy(st)
    st2.positions = positions_new
    disps2 = convert_positions_to_disps(st2, st)
    np.testing.assert_allclose(disps, disps2)


def test_get_structures_from_displacements(structure_rocksalt):
    """Test for get_structures_from_displacements."""
    st = structure_rocksalt
    shape = (2, 3, st.positions.shape[1])
    disps = np.ones(shape)
    disps[0] *= 0.01
    disps[1] *= 0.02
    (st1, st2) = get_structures_from_displacements(disps, structure_rocksalt)
    np.testing.assert_allclose(
        st1.positions,
        [
            [0.0025, 0.0025, 0.5025, 0.5025, 0.0025, 0.0025, 0.5025, 0.5025],
            [0.0025, 0.5025, 0.0025, 0.5025, 0.0025, 0.5025, 0.0025, 0.5025],
            [0.0025, 0.5025, 0.5025, 0.0025, 0.5025, 0.0025, 0.0025, 0.5025],
        ],
    )
    np.testing.assert_allclose(
        st2.positions,
        [
            [0.005, 0.005, 0.505, 0.505, 0.005, 0.005, 0.505, 0.505],
            [0.005, 0.505, 0.005, 0.505, 0.005, 0.505, 0.005, 0.505],
            [0.005, 0.505, 0.505, 0.005, 0.505, 0.005, 0.005, 0.505],
        ],
    )


def test_generate_random_const_displacements(structure_rocksalt):
    """Test generate_random_const_displacements."""
    st = structure_rocksalt
    disps, _ = generate_random_const_displacements(st, n_samples=3, displacements=0.01)
    assert disps.shape == (3, 3, 8)
