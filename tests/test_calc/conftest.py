"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pypolymlp.core.interface_vasp import PolymlpStructure, Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


@pytest.fixture(scope="session")
def unitcell_mlp_Al() -> (PolymlpStructure, str):
    """Return rocksalt structure."""
    unitcell = Poscar(path_file + "poscars/POSCAR.fcc.Al").structure
    pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
    return unitcell, pot
