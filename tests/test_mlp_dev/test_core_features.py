"""Tests of feature calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.mlp_dev.core.features import compute_features

cwd = Path(__file__).parent


def test_features(regdata_mp_149):
    """Tests of feature calculations."""
    params, datasets = regdata_mp_149
    features = compute_features(params, datasets)
    assert features.x.shape[0] == 35820
    assert features.x.shape[1] == 168
    assert features.n_data == (180, 34560, 180 * 6)
    assert features.first_indices[0] == (0, 1260, 180)
    assert features.cumulative_n_features is None

    x = features.x
    assert np.sum(x) == pytest.approx(5165294.450079148, rel=1e-6)
    assert np.sum(x[:, :20]) == pytest.approx(447.7438322711305, rel=1e-6)
    assert np.sum(x[:, 20:40]) == pytest.approx(15803.28147846774, rel=1e-6)
    assert np.sum(x[:, 40:60]) == pytest.approx(93568.30539681061, rel=1e-6)
    assert np.sum(x[:, 60:80]) == pytest.approx(308321.82717270905, rel=1e-6)
    assert np.sum(x[:, -60:-40]) == pytest.approx(1987480.894857876, rel=1e-6)
    assert np.sum(x[:, -40:-20]) == pytest.approx(34372.80650408206, rel=1e-6)
    assert np.sum(x[:, -20:]) == pytest.approx(2008586.8132866116, rel=1e-6)


def test_features_hybrid(regdata_mp_149):
    """Tests of feature calculations using a hybrid model."""
    params, datasets = regdata_mp_149
    params.type_indices = [0]
    params.type_full = True

    hybrid = params.as_hybrid_model()
    features = compute_features(hybrid, datasets)
    assert features.x.shape[0] == 35820
    assert features.x.shape[1] == 336
    assert features.n_data == (180, 34560, 180 * 6)
    assert tuple(features.cumulative_n_features) == (168, 336)
    assert tuple(features.first_indices[0]) == (0, 1260, 180)

    x = features.x
    assert np.sum(x) == pytest.approx(10330588.900158295, rel=1e-6)
    assert np.sum(x[:, :40]) == pytest.approx(16251.02531073887, rel=1e-6)
    assert np.sum(x[:, 40:80]) == pytest.approx(401890.1325695197, rel=1e-6)
    assert np.sum(x[:, 80:120]) == pytest.approx(717054.2979810806, rel=1e-6)
    assert np.sum(x[:, 120:160]) == pytest.approx(2804300.094916472, rel=1e-6)
    assert np.sum(x[:, 160:200]) == pytest.approx(1242982.1653808185, rel=1e-6)
    assert np.sum(x[:, 200:240]) == pytest.approx(393663.7339174809, rel=1e-6)
    assert np.sum(x[:, 240:]) == pytest.approx(4754447.450082185, rel=1e-6)
