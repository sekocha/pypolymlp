"""Tests of ASE integrator."""

from pathlib import Path

from pypolymlp.calculator.md.ase_md import IntegratorASE
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator
from pypolymlp.calculator.utils.ase_calculator_ref import (
    PolymlpFC2ASECalculator,
    PolymlpGeneralRefASECalculator,
    PolymlpRefASECalculator,
)
from pypolymlp.calculator.utils.ase_utils import (  # ase_atoms_to_structure,
    structure_to_ase_atoms,
)
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5

# import numpy as np
# import pytest


cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


fc2hdf5 = path_file + "others/fc2_Al_111.hdf5"
fc2 = load_fc2_hdf5(fc2hdf5)


def test_IntegratorASE_PolymlpASECalculator(unitcell_mlp_Al):
    """Test IntegratorASE using PolymlpASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    atoms_fcc = structure_to_ase_atoms(unitcell)
    calc = PolymlpASECalculator(pot=pot)

    itg = IntegratorASE(atoms_fcc, calc)
    assert not itg._use_reference
    assert not itg._use_fc2


def test_IntegratorASE_PolymlpFC2ASECalculator(unitcell_mlp_Al):
    """Test IntegratorASE using PolymlpFC2ASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    atoms_fcc = structure_to_ase_atoms(unitcell)
    calc = PolymlpFC2ASECalculator(fc2, unitcell, pot=pot, alpha=0.5)

    itg = IntegratorASE(atoms_fcc, calc)
    assert itg._use_reference
    assert itg._use_fc2


def test_IntegratorASE_PolymlpRefASECalculator(unitcell_mlp_Al):
    """Test IntegratorASE using PolymlpFC2ASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    atoms_fcc = structure_to_ase_atoms(unitcell)
    calc = PolymlpRefASECalculator(pot=pot, pot_ref=pot, alpha=0.5)

    itg = IntegratorASE(atoms_fcc, calc)
    assert itg._use_reference
    assert not itg._use_fc2


def test_IntegratorASE_PolymlpGeneralRefASECalculator(unitcell_mlp_Al):
    """Test IntegratorASE using PolymlpGeneralRefASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    atoms_fcc = structure_to_ase_atoms(unitcell)
    calc = PolymlpGeneralRefASECalculator(
        fc2,
        unitcell,
        pot_final=pot,
        pot_ref=pot,
        alpha=0.5,
    )

    itg = IntegratorASE(atoms_fcc, calc)
    assert itg._use_reference
    assert itg._use_fc2
