"""Tests of feature calculations using API."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscars = [
    path_file + "poscars/POSCAR-00001.MgO",
    path_file + "poscars/POSCAR-00002.MgO",
]


def test_features_from_polymlp1():
    """Test feature calculations from polymlp using API."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(features_force=False, features_stress=False)
    x = polymlp.features
    assert x.shape == (2, 324)
    assert np.sum(x) == pytest.approx(997193.0146734761, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-6.428600229143143, rel=1e-6)


def test_features_from_polymlp2():
    """Test feature calculations from polymlp using API."""
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(features_force=False, features_stress=False)
    x = polymlp.features
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)


def test_features_from_infile1():
    """Test feature calculations from input file using API."""
    infile = path_file + "mlps/polymlp.in.gtinv.MgO"
    polymlp = PypolymlpCalc(verbose=True, require_mlp=False)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(
        develop_infile=infile, features_force=False, features_stress=False
    )
    x = polymlp.features
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)
