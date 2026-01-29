"""Tests of PolymlpDataMLP."""

from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP

cwd = Path(__file__).parent


def test_data_mlp(regdata_mp_149):
    """Test for PolymlpDataMLP."""
    params, _ = regdata_mp_149

    n_features = 168
    coeffs = np.random.random(n_features)
    scales = np.random.random(n_features)

    data = PolymlpDataMLP(
        coeffs=coeffs,
        scales=scales,
        params=params,
    )

    coeffs = np.random.random(n_features * 2)
    scales = np.random.random(n_features * 2)
    hybrid_params = [params, params]

    data = PolymlpDataMLP(
        coeffs=coeffs,
        scales=scales,
        params=hybrid_params,
        cumulative_n_features=(168, 336),
    )
    coeffs_hybrid = data.hybrid_division(coeffs)
    assert len(coeffs_hybrid[0]) == 168
    assert len(coeffs_hybrid[1]) == 168
