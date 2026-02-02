"""Tests of phonon calculations using API."""

import os
import shutil
from pathlib import Path

from test_compute_phonon import _assert_phonon, _assert_qha

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.RS.idealMgO"
pot = path_file + "mlps/polymlp.yaml.pair.MgO"


def test_phonon_MgO():
    """Test phonon calculations from polymlp using API."""
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscar)
    polymlp.init_phonon(supercell_matrix=(2, 2, 2))
    polymlp.run_phonon()
    _assert_phonon(polymlp._phonon)

    polymlp.write_phonon()
    os.remove("phonon_mesh_qpoints.txt")
    os.remove("phonon_thermal_properties.yaml")
    os.remove("phonon_total_dos.dat")


def test_qha_MgO():
    """Test QHA calculations from polymlp using API."""
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscar)
    polymlp.run_qha(supercell_matrix=(1, 1, 1))
    _assert_qha(polymlp._qha)

    polymlp.write_qha()
    shutil.rmtree("polymlp_phonon_qha")
