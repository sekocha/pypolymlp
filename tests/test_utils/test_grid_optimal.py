"""Tests of grid search of polymlps."""

import glob
import os
from pathlib import Path

import numpy as np

from pypolymlp.utils.grid_optimal import find_optimal_mlps

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/grid-Ti/"
polymlp_dirs = glob.glob(path_file + "polymlp-*")
key = "test_vasprun_low"


def test_find_optimal_mlps():
    """Test find_optimal_mlps."""
    data_all, data_convex, system = find_optimal_mlps(
        polymlp_dirs,
        key=key,
        filename_all="tmp_all.yaml",
        filename_convex="tmp_convex.yaml",
    )
    assert system is None
    assert data_all.shape == (285, 6)
    assert data_convex.shape == (11, 6)
    polymlps = data_convex[:, 4]

    true = [
        "polymlp-00035",
        "polymlp-00064",
        "polymlp-00067",
        "polymlp-00068",
        "polymlp-00097",
        "polymlp-00074",
        "polymlp-00103",
        "polymlp-00076",
        "polymlp-00105",
        "polymlp-00281",
        "polymlp-00284",
    ]
    np.testing.assert_equal(polymlps, true)
    os.remove("tmp_all.yaml")
    os.remove("tmp_convex.yaml")
