"""Tests of utility functions for scale calculations."""

from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.core.utils_scales import compute_scales, round_scales

cwd = Path(__file__).parent


def test_compute_scales(dataxy_xtx_xty_mp_149):
    """Test for compute_scales using xtx and xty."""
    xe_sum = dataxy_xtx_xty_mp_149.xe_sum
    xe_sq_sum = dataxy_xtx_xty_mp_149.xe_sq_sum
    n_data = 180
    scales, zero_ids = compute_scales(
        scales=None,
        xe_sum=xe_sum,
        xe_sq_sum=xe_sq_sum,
        n_data=n_data,
        include_force=True,
    )
    true = [3.10570383e-03, 1.12433037e-05, 7.59175439e-04]
    np.testing.assert_allclose(scales[50:53], true)
    np.testing.assert_equal(zero_ids, False)


def test_round_scales():
    """Test for round_scales."""
    scales = np.array([0.5, 1e-11, 1e-20])
    scales, zero_ids = round_scales(scales, include_force=True)
    np.testing.assert_allclose(scales, [0.5, 1.0, 1.0])
    np.testing.assert_equal(zero_ids, [False, True, True])

    scales = np.array([0.5, 1e-11, 1e-20])
    scales, zero_ids = round_scales(scales, include_force=False)
    np.testing.assert_allclose(scales, [0.5, 1e-11, 1.0])
    np.testing.assert_equal(zero_ids, [False, False, True])
