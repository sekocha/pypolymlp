"""Tests of eigenvalue problem solver."""

import numpy as np

from pypolymlp.polyinv.eig_solver import eigh


def test_eigh():
    """Test eigh."""
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    eigvecs = eigh(p)
    np.testing.assert_allclose(eigvecs @ eigvecs.T, p)

    eigvecs = eigh(p, size_threshold=0)
    np.testing.assert_allclose(eigvecs @ eigvecs.T, p)
