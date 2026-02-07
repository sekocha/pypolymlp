"""Tests of ASE integrator."""

import os
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

    itg.set_integrator_Nose_Hoover_NVT(temperature=700)
    itg.activate_loggers(
        logfile="tmp.dat",
        interval_log=1,
        interval_save_forces=1,
        interval_save_trajectory=1,
    )
    itg.write_conditions()
    itg.activate_standard_output(interval=1)
    itg.run(n_eq=2, n_steps=3)

    assert itg.atoms is not None
    assert isinstance(itg.calculator, PolymlpASECalculator)

    assert itg.energies.shape[0] == 6
    assert itg.forces.shape == (6, 4, 3)
    assert len(itg.trajectory) == 6
    assert itg.static_energy is None

    assert itg.heat_capacity is not None
    assert itg.heat_capacity_eV is not None
    assert itg.average_energy is not None
    assert itg.average_total_energy is not None

    assert itg.average_displacement is None
    assert itg.delta_energies_10 is None
    assert itg.delta_energies_1a is None
    assert itg.average_delta_energy_10 is None
    assert itg.average_delta_energy_1a is None
    assert itg.free_energy_perturb is None

    itg.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")


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
