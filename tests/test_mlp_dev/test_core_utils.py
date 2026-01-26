"""Tests of utility functions for mlp development."""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.utils import (
    check_memory_size_in_regression,
    get_min_energy,
    set_params,
)

cwd = Path(__file__).parent


def test_set_params(regdata_mp_149):
    """Test for set_params."""
    params, _ = regdata_mp_149
    p, c, h = set_params(params)
    assert p == params
    assert c == params
    assert h is None
    p, c, h = set_params([params, params])
    assert p == h
    assert c.include_force == True


def test_get_min_energy(regdata_mp_149):
    """Test for get_min_energy."""
    _, datasets = regdata_mp_149
    min_e = get_min_energy(datasets)
    assert min_e == pytest.approx(-5.737324395625)


def test_check_memory_size_in_regression():
    """Test for check_memory_size_in_regression."""
    mem = check_memory_size_in_regression(20000, use_gradient=False)
    assert mem == pytest.approx(6.4)
    mem = check_memory_size_in_regression(20000, use_gradient=True)
    assert mem == pytest.approx(3.5)


#    first_indices = dataxy_mp_149.first_indices[0]
#    x = copy.deepcopy(dataxy_mp_149.x)
#    y = np.zeros(x.shape[0])
#    w = np.ones(x.shape[0])
#    x, y, w = apply_weights(x, y, w, datasets[0], first_indices)
#    np.testing.assert_allclose(y[:3], [-3.67121086e+02,-3.67115343e+02,-3.67102616e+02])
#    assert np.count_nonzero(np.isclose(w, 1.0)) == 34740
