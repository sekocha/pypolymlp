"""Tests of calculating feature attributes."""

from pathlib import Path

import numpy as np

from pypolymlp.mlp_dev.core.features_attr import get_features_attr, get_num_features

cwd = Path(__file__).parent


def test_get_num_features(regdata_mp_149):
    """Test for get_num_features."""
    params, _ = regdata_mp_149
    n_features = get_num_features(params)
    assert n_features == 168
    n_features = get_num_features([params, params, params])
    assert n_features == 168 * 3


def test_get_num_features_binary(params_MgO):
    """Test for get_num_features in binary system."""
    n_features = get_num_features(params_MgO)
    assert n_features == 1899


def test_get_features_attr(regdata_mp_149):
    """Test for get_features_attr."""
    params, _ = regdata_mp_149
    features_attr, polynomial_attr, atomtype_pair = get_features_attr(params)
    assert isinstance(features_attr, list) == True
    assert isinstance(polynomial_attr, list) == True
    assert isinstance(atomtype_pair, dict) == True
    assert features_attr[-1][0] == 6
    assert features_attr[-1][1] == 19
    assert features_attr[-1][2] == [0, 0, 0]
    np.testing.assert_equal(
        polynomial_attr,
        [
            [0, 0],
            [0, 20],
            [20, 20],
            [0, 40],
            [20, 40],
            [40, 40],
            [0, 60],
            [20, 60],
            [40, 60],
            [60, 60],
            [0, 80],
            [20, 80],
            [40, 80],
            [60, 80],
            [80, 80],
            [0, 100],
            [20, 100],
            [40, 100],
            [60, 100],
            [80, 100],
            [100, 100],
            [0, 120],
            [20, 120],
            [40, 120],
            [60, 120],
            [80, 120],
            [100, 120],
            [120, 120],
        ],
    )
    assert atomtype_pair[0][0] == [0, 0]


def test_get_features_attr_binary(params_MgO):
    """Test for get_features_attr in binary system."""
    features_attr, polynomial_attr, atomtype_pair = get_features_attr(params_MgO)
    assert isinstance(features_attr, list) == True
    assert isinstance(polynomial_attr, list) == True
    assert isinstance(atomtype_pair, dict) == True

    assert features_attr[-1][0] == 8
    assert features_attr[-1][1] == 19
    assert features_attr[-1][2] == [2, 2, 2]
    assert features_attr[-2][0] == 8
    assert features_attr[-2][1] == 19
    assert features_attr[-2][2] == [1, 2, 2]

    np.testing.assert_equal(
        polynomial_attr[6:9],
        [[1, 178], [178, 178], [0, 179]],
    )
    assert atomtype_pair[0][0] == [0, 0]
    assert atomtype_pair[1][0] == [0, 1]
    assert atomtype_pair[1][1] == [1, 0]
    assert atomtype_pair[2][0] == [1, 1]
