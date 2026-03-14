"""Pytest conftest.py."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


@pytest.fixture(scope="session")
def unitcell_mlp_Al() -> (PolymlpStructure, str):
    """Return fcc structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.fcc.Al"
    pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def unitcell_pair_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def unitcell_gtinv_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def unitcell_disp_pair_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.MgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def unitcell_disp_gtinv_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.MgO"
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def unitcell_wz_AlN() -> (PolymlpStructure, str):
    poscar = path_file + "poscars/POSCAR.WZ.AlN"
    pot = path_file + "mlps/polymlp.lammps.gtinv.AlN"
    unitcell = Poscar(poscar).structure
    prop = Properties(pot=pot)
    return unitcell, pot, prop


@pytest.fixture(scope="session")
def properties_Ag() -> Properties:
    pot = path_file + "mlps/polymlp.lammps.pair.Ag"
    prop = Properties(pot=pot)
    return prop


@pytest.fixture(scope="session")
def properties_TiAl() -> Properties:
    pot = path_file + "mlps/polymlp.lammps.gtinv.Ti-Al"
    prop = Properties(pot=pot)
    return prop


@pytest.fixture(scope="session")
def prototypes_Ag():
    api = PypolymlpAutoCalc(
        pot=path_file + "mlps/polymlp.lammps.pair.Ag",
        path_output="tmp",
    )
    api.load_prototypes()
    api.prototypes = api.prototypes[0:2]
    api.calc_prototypes()
    prototypes = api.prototypes
    shutil.rmtree("tmp")
    return prototypes
