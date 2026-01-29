"""Tests of utility functions for sequential regression."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.core.utils_sequential import (
    estimate_peak_memory,
    get_auto_batch_size,
    get_batch_slice,
    sum_array,
    sum_large_xtx,
    sum_xtx,
    symmetrize_xtx,
)

cwd = Path(__file__).parent


def test_estimate_peak_memory():
    """Test estimate_peak_memory."""
    mem = estimate_peak_memory(
        n_data=10000, n_features=20000, n_features_threshold=10000
    )
    assert mem == pytest.approx(6.4)
    mem = estimate_peak_memory(
        n_data=10000, n_features=20000, n_features_threshold=30000
    )
    assert mem == pytest.approx(8.0)


def test_get_batch_slice():
    """Test get_batch_slice."""
    begin, end = get_batch_slice(1000, batch_size=200)
    assert begin == [0, 200, 400, 600, 800]
    assert end == [200, 400, 600, 800, 1000]

    begin, end = get_batch_slice(1000, batch_size=150)
    assert begin == [0, 150, 300, 450, 600, 750, 900]
    assert end == [150, 300, 450, 600, 750, 900, 1000]


def test_get_auto_batch_size():
    """Test get_auto_batch_size."""
    batch_size = get_auto_batch_size(n_features=100)
    assert isinstance(batch_size, int) == True


def test_sum_array():
    """Test sum_array."""
    array1 = None
    array2 = np.random.random((3, 3))
    np.testing.assert_allclose(sum_array(array1, array2), array2)

    array1 = np.random.random((3, 3))
    array2 = np.random.random((3, 3))
    true = array1 + array2
    np.testing.assert_allclose(sum_array(array1, array2), true)


def test_sum_large_xtx():
    """Test sum_large_xtx."""
    xtx = np.random.random((3, 3))
    x = np.random.random((5, 3))
    true = xtx + x.T @ x
    np.testing.assert_allclose(sum_large_xtx(xtx, x), true)


def test_sum_xtx():
    """Test sum_xtx."""
    xtx = np.random.random((3, 3))
    x = np.random.random((5, 3))
    true = xtx + x.T @ x
    np.testing.assert_allclose(sum_xtx(xtx, x), true)


def test_symmetrize_xtx():
    """Test symmetrize_xtx."""
    mat = np.random.random((10, 100))
    x = mat.T @ mat
    xtx = np.zeros((200, 200))
    xtx[:100, :100] = x
    xtx[:100, 100:] = x
    xtx[100:, 100:] = x
    xtx = symmetrize_xtx(xtx, n_batch=2)

    true = np.zeros((200, 200))
    true[:100, :100] = x
    true[:100, 100:] = x
    true[100:, :100] = x
    true[100:, 100:] = x

    np.testing.assert_allclose(xtx, true)
