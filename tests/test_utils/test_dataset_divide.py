"""Tests of functions used for dividing dataset automatically."""

import glob
import shutil
from pathlib import Path

from pypolymlp.utils.dataset_divide import auto_divide_vaspruns

cwd = Path(__file__).parent


def test_auto_divide_vaspruns():
    """Test auto_divide_vaspruns."""
    path = str(cwd) + "/../test_mlp_dev_api/data-vasp-MgO/vaspruns/test1/"
    vaspruns = sorted(glob.glob(path + "vasprun-*.xml.*"))
    elements = ["Mg", "O"]
    auto_divide_vaspruns(
        vaspruns,
        elements,
        path_output="./tmp",
        n_divide=3,
        verbose=True,
    )
    shutil.rmtree("tmp")
