"""Tests of rotation utils."""

import numpy as np
import pytest

from pypolymlp.calculator.utils.rotation_utils import (
    recover_rotated_forces,
    recover_rotated_stress,
    triangularize_axis,
)


def test_triangularize_axis():
    """Test triangularize_axis."""
    axis = np.array([[3.0, 0.2, 0.3], [0.1, 3.5, -0.1], [0.1, 0.05, 3.7]])
    tri_axis, tri_axis_full, rotation = triangularize_axis(axis)

    true = [3.00333148, 3.4916169, 3.68908242, 0.31798022, -0.06827884, 0.41953411]
    np.testing.assert_allclose(tri_axis, true, rtol=1e-5)

    true_mat = [[true[0], true[3], true[5]], [0, true[1], true[4]], [0, 0, true[2]]]
    np.testing.assert_allclose(tri_axis_full, true_mat, rtol=1e-5)

    true_rot = np.array(
        [
            [0.99889074, 0.03329636, 0.03329636],
            [-0.03368855, 0.99936864, 0.01128773],
            [-0.0328995, -0.01239691, 0.99938178],
        ]
    )
    np.testing.assert_allclose(rotation, true_rot, rtol=1e-5)

    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 0] @ rotation[:, 0] == pytest.approx(1.0)
    assert rotation[:, 1] @ rotation[:, 1] == pytest.approx(1.0)
    assert rotation[:, 2] @ rotation[:, 2] == pytest.approx(1.0)
    assert rotation[:, 0] @ rotation[:, 1] == pytest.approx(0.0)
    assert rotation[:, 1] @ rotation[:, 2] == pytest.approx(0.0)
    assert rotation[:, 2] @ rotation[:, 0] == pytest.approx(0.0)


def test_recover_rotated_forces():
    """Test recover_rotated_forces."""
    axis = np.array([[3.0, 0.2, 0.3], [0.1, 3.5, -0.1], [0.1, 0.05, 3.7]])
    _, _, rotation = triangularize_axis(axis)

    forces = np.random.random((3, 8))
    recovered = recover_rotated_forces(forces, rotation)
    np.testing.assert_allclose(recovered, rotation.T @ forces)


def test_recover_rotated_stress():
    """Test recover_rotated_stress."""
    axis = np.array([[3.0, 0.2, 0.3], [0.1, 3.5, -0.1], [0.1, 0.05, 3.7]])
    _, _, rotation = triangularize_axis(axis)
    stress = np.random.random((3, 3))
    recovered = recover_rotated_stress(stress, rotation)
    np.testing.assert_allclose(recovered, rotation.T @ stress @ rotation)
