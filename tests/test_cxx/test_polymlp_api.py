"""Tests of wrapper class for PolymlpAPI."""

from pathlib import Path

from pypolymlp.cxx_wrapper.wrapper import PolymlpCPPAPI

# import numpy as np
# import pytest


cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"


def test_cpp_api():
    """Test for convert_unit."""
    api = PolymlpCPPAPI()
    api.parse_polymlp_file(path_files + "polymlp.yaml.MgO", ["Mg", "O"], [1.0, 1.0])

    fp = api.feature_params
    print(fp.cutoff)
    print(fp.pair_type)
    api.convert_unit(energy_conv=3.0, length_conv=0.5, inv_length_conv=2.0)
    assert 1 == 0
