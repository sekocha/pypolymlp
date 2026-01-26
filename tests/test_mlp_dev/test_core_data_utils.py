"""Tests of PolymlpDataXY."""

from pathlib import Path

import pytest

cwd = Path(__file__).parent


def test_dataxy(dataxy_mp_149):
    """Test for test_dataxy."""
    n_samples = 10
    total_n_atoms = [64 for i in range(n_samples)]
    x, y = dataxy_mp_149.slice(n_samples, total_n_atoms)
    assert x.shape == (1990, 168)
    assert y.shape[0] == 1990

    assert dataxy_mp_149.x is not None
    assert dataxy_mp_149.y is not None
    assert dataxy_mp_149.xtx is None
    assert dataxy_mp_149.xty is None
    assert dataxy_mp_149.scales is not None
    assert dataxy_mp_149.min_energy is not None

    assert dataxy_mp_149.first_indices[0] == (0, 1260, 180)
    assert dataxy_mp_149.cumulative_n_features is None
    assert dataxy_mp_149.total_n_data == 0
    assert dataxy_mp_149.n_structures == 180

    assert dataxy_mp_149.xe_sum is None
    assert dataxy_mp_149.xe_sq_sum is None
    assert dataxy_mp_149.y_sq_norm == 0.0


def test_dataxy_xtx_xty(dataxy_xtx_xty_mp_149):
    """Test for test_dataxy using xtx and xty."""
    assert dataxy_xtx_xty_mp_149.x is None
    assert dataxy_xtx_xty_mp_149.y is None
    assert dataxy_xtx_xty_mp_149.xtx is not None
    assert dataxy_xtx_xty_mp_149.xty is not None
    assert dataxy_xtx_xty_mp_149.scales is not None
    assert dataxy_xtx_xty_mp_149.min_energy is not None

    assert dataxy_xtx_xty_mp_149.first_indices is None
    assert dataxy_xtx_xty_mp_149.cumulative_n_features is None
    assert dataxy_xtx_xty_mp_149.total_n_data == 35820
    assert dataxy_xtx_xty_mp_149.n_structures == 0

    assert dataxy_xtx_xty_mp_149.xe_sum is not None
    assert dataxy_xtx_xty_mp_149.xe_sq_sum is not None
    assert dataxy_xtx_xty_mp_149.y_sq_norm == pytest.approx(24262983.5578237)
