"""Tests of grid_enum.py."""

import pytest

from pypolymlp.utils.grid_search.api_grid_search import PolymlpGridSearch
from pypolymlp.utils.grid_search.grid_enum import enum_gtinv_models, enum_pair_models


def test_enum_pair_model():
    """Test enum_pair_model."""
    elements = ["Be"]
    grid1 = PolymlpGridSearch(elements=elements, verbose=True)
    grid1.set_params()
    params_grid = grid1._grid

    params = enum_pair_models(params_grid, elements)
    assert len(params) == 8
    assert params[7].model.cutoff == pytest.approx(6.0)
    assert params[7].model.n_gaussians == pytest.approx(9)


def test_enum_gtinv_model():
    """Test enum_gtinv_models."""
    elements = ["Be"]
    grid1 = PolymlpGridSearch(elements=elements, verbose=True)
    grid1.set_params()
    params_grid = grid1._grid

    params = enum_gtinv_models(params_grid, elements)
    assert len(params) == 104
    assert params[7].model.cutoff == pytest.approx(5.0)
    assert params[7].model.n_gaussians == pytest.approx(6)
