"""Tests of functions to draw figures for properties."""

import shutil
from pathlib import Path

from pypolymlp.calculator.auto.autocalc_prototypes import AutoCalcPrototypes

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_AutoCalcPrototypes1(properties_Ag):
    """Test AutoCalcPrototypes for elemental system."""
    api = AutoCalcPrototypes(properties_Ag, path_output="tmp")
    api.load_structures()
    assert len(api.prototypes) == 18
    api.prototypes = api.prototypes[0:3]
    assert len(api.prototypes) == 3

    vaspruns = [
        path_file + "others/vasprun-00001-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
    ]
    api.set_dft_properties(vaspruns, ["104296", "105489", "105489"])
    api.run()
    api.save_properties()
    shutil.rmtree("tmp")


def test_AutoCalcPrototypes2(properties_TiAl):
    """Test AutoCalcPrototypes for binary alloy system."""
    api = AutoCalcPrototypes(properties_TiAl, path_output="tmp")
    api.load_structures()
    assert len(api.prototypes) == 69
    api.prototypes = api.prototypes[0:3]
    assert len(api.prototypes) == 3

    vaspruns = [
        path_file + "others/vasprun-00001-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
    ]
    api.set_dft_properties(vaspruns, ["104296", "105489", "105489"])
    api.run()
    api.save_properties()
    shutil.rmtree("tmp")
