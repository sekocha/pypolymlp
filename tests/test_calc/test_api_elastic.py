"""Tests of elastic constant calculations using API."""

import os
from pathlib import Path

from test_compute_elastic import _assert_elastic_gtinv_MgO, _assert_elastic_pair_MgO

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_pair_MgO():
    """Test elastic constants with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.run_elastic_constants(poscar=poscar)
    _assert_elastic_pair_MgO(polymlp.elastic_constants)

    polymlp.write_elastic_constants(filename="tmp.yaml")
    os.remove("tmp.yaml")


def test_gtinv_MgO():
    """Test elastic constants with polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.run_elastic_constants(poscar=poscar)
    _assert_elastic_gtinv_MgO(polymlp.elastic_constants)

    polymlp.write_elastic_constants(filename="tmp.yaml")
    os.remove("tmp.yaml")
