"""Tests of feature calculations using API."""

from pathlib import Path

from test_compute_features import _assert_gtinv_MgO, _assert_pair_MgO

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
    _assert_pair_MgO(x)


def test_features_from_polymlp2():
    """Test feature calculations from polymlp using API."""
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(features_force=False, features_stress=False)
    x = polymlp.features
    _assert_gtinv_MgO(x)


def test_features_from_infile1():
    """Test feature calculations from input file using API."""
    infile = path_file + "mlps/polymlp.in.gtinv.MgO"
    polymlp = PypolymlpCalc(verbose=True, require_mlp=False)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(
        develop_infile=infile, features_force=False, features_stress=False
    )
    x = polymlp.features
    _assert_gtinv_MgO(x)
