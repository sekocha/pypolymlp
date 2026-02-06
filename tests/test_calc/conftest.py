"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


@pytest.fixture(scope="session")
def unitcell_mlp_Al() -> (PolymlpStructure, str):
    """Return fcc structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.fcc.Al"
    pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
    unitcell = Poscar(poscar).structure
    return unitcell, pot


@pytest.fixture(scope="session")
def unitcell_pair_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(poscar).structure
    return unitcell, pot


@pytest.fixture(scope="session")
def unitcell_gtinv_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    unitcell = Poscar(poscar).structure
    return unitcell, pot


@pytest.fixture(scope="session")
def unitcell_disp_pair_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.MgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(poscar).structure
    return unitcell, pot


@pytest.fixture(scope="session")
def unitcell_disp_gtinv_MgO() -> (PolymlpStructure, str):
    """Return rocksalt structure and potential file name."""
    poscar = path_file + "poscars/POSCAR.RS.MgO"
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    unitcell = Poscar(poscar).structure
    return unitcell, pot


@pytest.fixture(scope="session")
def unitcell_wz_AlN() -> (PolymlpStructure, str):
    poscar = path_file + "poscars/POSCAR.WZ.AlN"
    pot = path_file + "mlps/polymlp.lammps.gtinv.AlN"
    unitcell = Poscar(poscar).structure
    return unitcell, pot
