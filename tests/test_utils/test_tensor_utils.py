"""Tests of tensor utility functions."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.utils.tensor_utils_O2 import (
    compute_perm_projector_O2,
    compute_projector_O2,
    compute_spg_projector_O2,
    compute_tensor_basis_O2,
)
from pypolymlp.utils.tensor_utils_O4 import (
    compute_perm_projector_O4,
    compute_projector_O4,
    compute_spg_projector_O4,
    compute_tensor_basis_O4,
)

cwd = Path(__file__).parent


def test_projector_O2(structure_rocksalt):
    proj = compute_projector_O2(structure_rocksalt)
    assert proj.shape == (9, 9)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 9
    assert np.trace(proj) == pytest.approx(1.0)


def test_compute_spg_projector_O2(structure_rocksalt):
    """Test compute_spg_projector_O2."""
    proj = compute_spg_projector_O2(structure_rocksalt)
    assert proj.shape == (9, 9)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 9
    assert np.trace(proj) == pytest.approx(1.0)


def test_compute_perm_projector_O2(structure_rocksalt):
    """Test compute_perm_projector_O2."""
    proj = compute_perm_projector_O2(structure_rocksalt)
    assert proj.shape == (9, 9)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 15
    assert np.trace(proj) == pytest.approx(6)


def test_compute_tensor_basis_O2(structure_rocksalt):
    """Test compute_tensor_basis_O2."""
    basis = compute_tensor_basis_O2(structure_rocksalt)
    assert basis.shape == (9, 1)
    assert np.count_nonzero(np.abs(basis) > 1e-12) == 3

    proj_true = compute_projector_O2(structure_rocksalt)
    np.testing.assert_allclose(basis @ basis.T, proj_true, atol=1e-8)


def test_projector_O4(structure_rocksalt):
    proj = compute_projector_O4(structure_rocksalt)
    assert proj.shape == (81, 81)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 189
    assert np.trace(proj) == pytest.approx(3)


def test_compute_spg_projector_O4(structure_rocksalt):
    """Test compute_spg_projector_O4."""
    proj = compute_spg_projector_O4(structure_rocksalt)
    assert proj.shape == (81, 81)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 117
    assert np.trace(proj) == pytest.approx(4)


def test_compute_perm_projector_O4(structure_rocksalt):
    """Test compute_perm_projector_O4."""
    proj = compute_perm_projector_O4(structure_rocksalt)
    assert proj.shape == (81, 81)
    assert np.count_nonzero(np.abs(proj) > 1e-12) == 399
    assert np.trace(proj) == pytest.approx(21)


def test_compute_tensor_basis_O4(structure_rocksalt):
    """Test compute_tensor_basis_O4."""
    basis = compute_tensor_basis_O4(structure_rocksalt)
    assert basis.shape == (81, 3)
    assert np.count_nonzero(np.abs(basis) > 1e-12) == 21

    proj_true = compute_projector_O4(structure_rocksalt)
    np.testing.assert_allclose(basis @ basis.T, proj_true, atol=1e-8)
