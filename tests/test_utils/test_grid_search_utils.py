"""Tests of grid_utils."""

from pypolymlp.utils.grid_search.api_grid_search import PolymlpGridSearch
from pypolymlp.utils.grid_search.grid_utils import GaussianAttrs, GtinvAttrs


def test_GtinvAttrs():
    """Test GtinvAttrs."""
    _ = GtinvAttrs(model_type=2, order=3, max_l=[4, 4])


def test_GaussianAttrs():
    """Test GaussianAttrs."""
    _ = GaussianAttrs(cutoff=6.0, n_gaussians=10)


def test_ParamsGrid_gtinv():
    """Test ParamsGrid for invariant parameter enumeration."""
    elements = ["Be"]
    grid1 = PolymlpGridSearch(elements=elements, verbose=True)
    grid1.set_params()
    params_grid = grid1._grid

    assert params_grid.get_model_types_pair() == [2]
    assert len(params_grid.gtinv_attrs) == 26


def test_ParamsGrid_pair():
    """Test ParamsGrid for pair parameter enumeration."""
    elements = ["Be"]
    grid1 = PolymlpGridSearch(elements=elements, verbose=True)
    grid1.set_params(gtinv=False, model_types=(2, 3, 4))
    params_grid = grid1._grid

    assert params_grid.get_model_types_pair() == [2]
