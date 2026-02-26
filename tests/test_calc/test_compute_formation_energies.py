"""Tests of functions for feature calculations."""

from pathlib import Path

from pypolymlp.calculator.compute_formation_energies import PolymlpFormationEnergies
from pypolymlp.core.interface_vasp import parse_structures_from_poscars

# import numpy as np
# import pytest


cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "mlps/polymlp.lammps.gtinv.Cu-Ag-Au"

poscars = [
    path_file + "poscars/POSCAR1.Cu-Ag-Au",
    path_file + "poscars/POSCAR2.Cu-Ag-Au",
    path_file + "poscars/POSCAR3.Cu-Ag-Au",
    path_file + "poscars/POSCAR4.Cu-Ag-Au",
    path_file + "poscars/POSCAR5.Cu-Ag-Au",
]


def test_PolymlpFormationEnergies():
    """Test PolymlpFormationEnergies."""
    structures = parse_structures_from_poscars(poscars)
    api = PolymlpFormationEnergies(pot=pot)
    api.compute(structures)
    assert 1 == 0
