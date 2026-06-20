"""Tests of wrapper class for PolymlpAPI."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.cxx.wrapper.wrapper import PolymlpCPPAPI

cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"


api_rep = PolymlpCPPAPI()
api_rep.parse_polymlp_file(path_files + "polymlp.yaml.MgO", ["Mg", "O"], [1.0, 1.0])
fp_rep = api_rep.feature_params


def test_cpp_api():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.parse_polymlp_file(path_files + "polymlp.yaml.MgO", ["Mg", "O"], [1.0, 1.0])
    fp = api.feature_params
    assert api.n_variables == 1899

    assert fp.cutoff == pytest.approx(8.0)
    api.convert_unit(energy_conv=3.0, length_conv=0.5, inv_length_conv=2.0)
    assert fp.cutoff == pytest.approx(4.0)


def test_cpp_api2():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.set_features(fp_rep)
    assert api.n_variables == 1899


def test_cpp_api3():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.set_model_parameters(fp_rep)


def test_cpp_api4():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    pot = np.ones(1899)
    api.set_potential_model(fp_rep, pot)
    assert api.n_variables == 1899
