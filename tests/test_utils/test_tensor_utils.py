"""Tests of tensor utility functions."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.tensor_utils import (
    compute_spg_projector_O2,
    compute_tensor_basis_O2,
)

cwd = Path(__file__).parent


def test_compute_spg_projector_O2(structure_rocksalt):
    """Test compute_spg_projector_O2."""
    proj = compute_spg_projector_O2(structure_rocksalt)
    assert proj.shape == (9, 9)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 9


def test_compute_tensor_basis_O2(structure_rocksalt):
    """Test compute_tensor_basis_O2."""
    basis = compute_tensor_basis_O2(structure_rocksalt)
    assert basis.shape == (9, 1)
    assert np.count_nonzero(np.abs(basis) > 1e-12) == 3
    assert basis[0, 0] == pytest.approx(1 / np.sqrt(3))
    assert basis[4, 0] == pytest.approx(1 / np.sqrt(3))
    assert basis[8, 0] == pytest.approx(1 / np.sqrt(3))
