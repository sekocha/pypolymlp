"""Tests of wrapper class for PolymlpAPI."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.cxx.wrapper.wrapper import PolymlpCPPAPI

cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"


api_rep = PolymlpCPPAPI()
api_rep.parse_polymlp_file(
    path_files + "polymlp.yaml.pair.MgO", ["Mg", "O"], [1.0, 1.0]
)
fp_rep = api_rep.feature_params
n_variables_MgO = 324


def test_cpp_api2():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.set_features(fp_rep)
    assert api.n_variables == n_variables_MgO


def test_cpp_api3():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.set_model_parameters(fp_rep)


def test_cpp_api4():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    pot = np.ones(n_variables_MgO)
    api.set_potential_model(fp_rep, pot)
    assert api.n_variables == n_variables_MgO


def test_compute_properties():
    """Test features and products."""

    antp = np.arange(n_variables_MgO)

    api = PolymlpCPPAPI()
    api.set_features(fp_rep)
    dn0 = api.compute_features_real(antp, 0)
    dn1 = api.compute_features_real(antp, 1)
    assert dn0.shape == (18,)
    assert dn1.shape == (18,)
    assert np.sum(dn0) == pytest.approx(153.0)
    assert np.sum(dn1) == pytest.approx(153.0)

    prod_e, prod_f = api_rep.compute_sum_of_prod_antp(antp, 0)
    assert prod_e.shape == (18,)
    assert prod_f.shape == (18,)
    assert np.sum(prod_e) == pytest.approx(823.0413669972496)
    assert np.sum(prod_f) == pytest.approx(1575.2007772518964)
