"""Tests of utility functions for mlp development."""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.utils import (
    check_memory_size_in_regression,
    collect_observations,
    get_min_energy,
    get_observation_slices,
)

cwd = Path(__file__).parent


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


def test_get_observation_slices(regdata_mp_149):
    """Test for get_observation_slices."""
    _, datasets = regdata_mp_149
    data = datasets[0]
    first_indices = (0, 1260, 180)
    slices = get_observation_slices(data, first_indices)
    assert slices == (0, 180, 1260, 35820, 180, 1260)


def test_collect_observations(regdata_mp_149):
    """Test collect_observations."""
    _, datasets = regdata_mp_149
    data = datasets[0]
    first_indices = (0, 1260, 180)
    y = collect_observations(data, n_data=35820, first_indices=first_indices)
    assert y[0] == pytest.approx(-367.12108579)
    assert y.shape == (35820,)
