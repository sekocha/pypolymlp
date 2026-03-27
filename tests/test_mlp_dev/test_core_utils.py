"""Tests of utility functions for mlp development."""

from pathlib import Path

import pytest

from pypolymlp.mlp_dev.core.utils import check_memory_size_in_regression, get_min_energy

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
