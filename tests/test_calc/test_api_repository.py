"""Tests of API for calculating properties posted for repository web."""

import glob
import shutil
from pathlib import Path

from pypolymlp.api.pypolymlp_repository import MLPAttr, PypolymlpRepository

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_run():
    """Test web content generation using PypolymlpRepository."""
    mlp_paths = [
        path_file + "others/repository/Ag-Au-2026-03-10/polymlps/mlp3",
        path_file + "others/repository/Ag-Au-2026-03-10/polymlps/mlp4",
        path_file + "others/repository/Ag-Au-2026-03-10/polymlps/mlp5",
    ]
    rep = PypolymlpRepository(mlp_paths=mlp_paths)
    rep.extract_convex_polymlps(key="dataset/vasprun-*.xml.polymlp")
    # TODO: calc_properties
    for path in glob.glob("Ag-Au*"):
        shutil.rmtree(path)


def test_web():
    """Test web content generation using PypolymlpRepository."""
    rep = PypolymlpRepository()
    rep.generate_web_contents(
        path_prediction=path_file + "others/repository/Ag-Au-2026-03-10"
    )
    for path in glob.glob("web-Ag-Au*"):
        shutil.rmtree(path)


def test_MLPAttr():
    """Test MLPAttr."""
    mlp_attr = MLPAttr(path_file + "others/repository/Ag-Au-2026-03-10", "mlp3")
    mlp_attr.set_autocalc()
    assert mlp_attr.autocalc._n_types == 2
