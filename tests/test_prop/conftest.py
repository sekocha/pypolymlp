"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


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
