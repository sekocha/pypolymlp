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
n_variables_MgO = 1899


def test_cpp_api():
    """Test PolymlpAPI."""
    api = PolymlpCPPAPI()
    api.parse_polymlp_file(path_files + "polymlp.yaml.MgO", ["Mg", "O"], [1.0, 1.0])
    fp = api.feature_params
    assert api.n_variables == n_variables_MgO

    assert fp.cutoff == pytest.approx(8.0)
    api.convert_unit(energy_conv=3.0, length_conv=0.5, inv_length_conv=2.0)
    assert fp.cutoff == pytest.approx(4.0)


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


def test_compute_anlmtp_conjugate():
    """Test compute_anlmtp_conjugate."""
    anlmtp_r = np.arange(n_variables_MgO)
    anlmtp_i = np.arange(n_variables_MgO)
    anlmtp0 = api_rep.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, 0)
    anlmtp1 = api_rep.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, 1)
    assert anlmtp0.shape == (450,)
    assert anlmtp1.shape == (450,)
    assert np.sum(anlmtp0) == (31581 + 17199j)
    assert np.sum(anlmtp1) == (31581 + 17199j)


def test_compute_properties():
    """Test features and products."""

    anlmtp_r = np.arange(n_variables_MgO)
    anlmtp_i = np.arange(n_variables_MgO)
    anlmtp0 = api_rep.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, 0)
    anlmtp1 = api_rep.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, 1)

    api = PolymlpCPPAPI()
    api.set_features(fp_rep)
    dn0 = api.compute_features(anlmtp0, 0)
    dn1 = api.compute_features(anlmtp1, 1)
    assert dn0.shape == (891,)
    assert dn1.shape == (891,)
    assert np.sum(dn0) == pytest.approx(849391612.5622855)
    assert np.sum(dn1) == pytest.approx(849391612.5622855)

    anlmtp_dfx = np.ones((450, 8))
    anlmtp_dfy = -np.ones((450, 8)) * 2
    anlmtp_dfy[200, 3] = 0.02
    anlmtp_dfz = np.ones((450, 8)) * 0.5
    anlmtp_dfz[100, 3] = -0.05
    anlmtp_ds = -np.ones((450, 6))
    anlmtp_ds[300, 3] = 0.03
    dfx, dfy, dfz, ds = api.compute_features_deriv(
        anlmtp0, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds, 0
    )
    assert dfx.shape == (891, 8)
    assert dfy.shape == (891, 8)
    assert dfz.shape == (891, 8)
    assert ds.shape == (891, 6)
    assert np.sum(dfx) == pytest.approx(-14716083.219391469)
    assert np.sum(dfy) == pytest.approx(29122027.72943048)
    assert np.sum(dfz) == pytest.approx(-7326280.333739502)
    assert np.sum(ds) == pytest.approx(10734296.686030138)

    prod_e, prod_f = api_rep.compute_sum_of_prod_anlmtp(anlmtp0, 0)
    assert prod_e.shape == (270,)
    assert prod_f.shape == (270,)
    assert np.sum(prod_e) == pytest.approx((-24266095.550829127 - 955759.7528607883j))
    assert np.sum(prod_f) == pytest.approx((-72763009.4963038 - 2902187.696305793j))
