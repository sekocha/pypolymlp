"""Tests of phonon calculations using API."""

from pathlib import Path

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

# from test_compute_phonon import _assert_phonon, _assert_qha


cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.RS.idealMgO"
pot = path_file + "mlps/polymlp.yaml.pair.MgO"


def test_phonon_MgO():
    """Test phonon calculations from polymlp using API."""
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscar)


def test_qha_MgO():
    """Test QHA calculations from polymlp using API."""
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscar)
