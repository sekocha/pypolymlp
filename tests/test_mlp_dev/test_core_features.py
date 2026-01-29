"""Tests of feature calculations."""

from pathlib import Path

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


def test_features_hybrid(regdata_mp_149):
    """Tests of feature calculations using a hybrid model."""
    params, datasets = regdata_mp_149
    params.type_indices = [0]
    params.type_full = True
    features = compute_features([params, params], datasets)
    assert features.x.shape[0] == 35820
    assert features.x.shape[1] == 336
    assert features.n_data == (180, 34560, 180 * 6)
    assert tuple(features.cumulative_n_features) == (168, 336)
    assert tuple(features.first_indices[0]) == (0, 1260, 180)
