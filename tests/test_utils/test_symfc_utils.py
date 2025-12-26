"""Tests of symfc utility functions."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.utils.symfc_utils import (
    construct_basis_cartesian,
    construct_basis_fractional_coordinates,
    structure_to_symfc_cell,
)

cwd = Path(__file__).parent


def test_structure_to_symfc_cell(structure_rocksalt):
    """Test structure converter."""
    cell = structure_to_symfc_cell(structure_rocksalt)
    assert len(cell.numbers) == 8
    assert cell.cell.shape == (3, 3)
    assert cell.scaled_positions.shape == (8, 3)


def test_basis(structure_rocksalt):
    """Test basis construction."""
    st = copy.deepcopy(structure_rocksalt)
    st.positions[:, 0] += 0.01

    basis_c = construct_basis_cartesian(st)
    proj_c = basis_c @ basis_c.T
    basis_f = construct_basis_fractional_coordinates(st)
    proj_f = basis_f @ basis_f.T

    assert proj_f.shape == (24, 24)
    np.testing.assert_allclose(proj_f, proj_c)
    np.testing.assert_allclose(proj_f[:3, :3], 0.29166667, atol=1e-6)
    np.testing.assert_allclose(
        proj_f[3:6, 3:6],
        [
            [0.29166667, -0.04166667, -0.04166667],
            [-0.04166667, 0.125, 0.125],
            [-0.04166667, 0.125, 0.125],
        ],
        atol=1e-6,
    )
