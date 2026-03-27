"""Tests of grid_io."""

import os

from pypolymlp.utils.grid_search.api_grid_search import PolymlpGridSearch
from pypolymlp.utils.grid_search.grid_io import save_params


def test_save_params():
    """Test save_params."""
    elements = ["Ag", "Au"]
    grid1 = PolymlpGridSearch(elements=elements, verbose=True)
    grid1.set_params()
    grid1.run()
    params = grid1.grid
    save_params(params[0], filename="tmp.in")
    os.remove("tmp.in")
