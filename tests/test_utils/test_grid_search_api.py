"""Tests of grid_search/api_grid_search.py."""

import shutil

import numpy as np

from pypolymlp.utils.grid_search.api_grid_search import PolymlpGridSearch


def test_grid_search1():
    """Test api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Be"], verbose=True)
    grid1.set_params()
    grid1.run()
    assert len(grid1.grid) == 112

    grid1.save_models(path="tmp")
    shutil.rmtree("tmp")


def test_grid_search2():
    """Test api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Ag", "Au"], verbose=True)
    grid1.set_params()
    grid1.run()
    assert len(grid1.grid) == 112

    grid1.save_models(path="tmp")
    shutil.rmtree("tmp")


def test_grid_search3():
    """Test api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Ag", "Au"], verbose=True)
    grid1.set_params()
    grid1.enum_pair_models()
    assert len(grid1.grid) == 8

    grid1.enum_gtinv_models()
    assert len(grid1.grid) == 112


def test_grid_search_input1():
    """Test api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Ag", "Au"], verbose=True)
    grid1.set_params(
        cutoffs=(6.0, 8.0, 10.0),
        nums_gaussians=(8, 10, 13),
        model_types=(2, 3, 4),
        gtinv=True,
        gtinv_order_ub=6,
        gtinv_maxl_ub=(12, 8, 2, 1, 1),
        gtinv_maxl_int=(4, 4, 2, 1, 1),
        include_force=True,
        include_stress=True,
        regression_alpha=(-4, 1, 6),
    )
    grid1.run()
    assert len(grid1.grid) == 459


def test_grid_search_input2():
    """Test api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Ag", "Au"], verbose=True)
    grid1.set_params(
        cutoffs=(6.0, 8.0, 10.0),
        nums_gaussians=(8, 10, 13),
        model_types=(2, 3, 4),
        gtinv=False,
        include_force=True,
        include_stress=True,
        regression_alpha=(-4, 1, 6),
    )
    grid1.run()
    assert len(grid1.grid) == 18


def test_grid_search_local_functions():
    """Test local functions in api_grid_search."""
    grid1 = PolymlpGridSearch(elements=["Be"], verbose=True)
    cutoffs = grid1._auto_cutoff()
    np.testing.assert_allclose(cutoffs, [5.0, 6.0])

    grid1 = PolymlpGridSearch(elements=["Ba"], verbose=True)
    cutoffs = grid1._auto_cutoff()
    np.testing.assert_allclose(cutoffs, [8.0, 11.0])

    radial_params = grid1._auto_gaussians(cutoffs, None)
    assert radial_params[0].cutoff == 8.0
    assert radial_params[1].cutoff == 8.0
    assert radial_params[2].cutoff == 11.0
    assert radial_params[3].cutoff == 11.0
    assert radial_params[0].n_gaussians == 9
    assert radial_params[1].n_gaussians == 12
    assert radial_params[2].n_gaussians == 12
    assert radial_params[3].n_gaussians == 16
