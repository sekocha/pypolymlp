"""Tests of ASE calculator with reference states using polymlp."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.utils.ase_calculator_ref import (
    PolymlpFC2ASECalculator,
    PolymlpGeneralRefASECalculator,
    PolymlpRefASECalculator,
    convert_atoms_to_str,
)
from pypolymlp.calculator.utils.ase_utils import (
    ase_atoms_to_structure,
    structure_to_ase_atoms,
)
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


st_fcc = Poscar(path_file + "poscars/POSCAR.fcc.Al").structure
atoms_fcc = structure_to_ase_atoms(st_fcc)

fc2hdf5 = path_file + "others/fc2_Al_111.hdf5"
fc2 = load_fc2_hdf5(fc2hdf5)


def test_convert_atoms_to_str():
    """Test convert_atoms_to_str."""
    structure = ase_atoms_to_structure(atoms_fcc)
    disps, _ = convert_atoms_to_str(atoms_fcc, structure)
    np.testing.assert_allclose(disps, 0.0)


def testPolymlpFC2ASECalculator(unitcell_mlp_Al):
    """Test PolymlpFC2ASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    calc = PolymlpFC2ASECalculator(fc2, unitcell, pot=pot, alpha=0.5)
    assert calc._use_reference
    assert calc._use_fc2
    assert calc.alpha == 0.5

    calc.calculate(atoms_fcc)
    assert calc.results["energy"] == pytest.approx(-13.71119226623514)
    assert np.sum(calc.results["forces"]) == pytest.approx(0.0)
    assert calc.delta_energy_10 == pytest.approx(0.0)
    assert calc.delta_energy_1a == pytest.approx(0.0)
    assert calc.average_displacement == pytest.approx(0.0, abs=1e-8)
    assert calc.static_energy == pytest.approx(-13.71119226623514)

    disps = np.array(
        [0.1, 0.2, 0.3, 0.01, 0.03, -0.01, 0.03, 0.05, 0.02, -0.1, 0.1, 0.1]
    )
    energy, forces = calc._eval_fc2_model(disps)
    assert energy == pytest.approx(-13.339222531849853)
    assert np.sum(forces) == pytest.approx(0.0)
    assert np.sum(np.abs(forces)) == pytest.approx(7.525484324213286)


def testPolymlpRefASECalculator(unitcell_mlp_Al):
    """Test PolymlpRefASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    calc = PolymlpRefASECalculator(pot=pot, pot_ref=pot, alpha=0.5)
    assert calc._use_reference
    assert not calc._use_fc2
    assert calc.alpha == 0.5
    calc.alpha = 0.5

    calc.calculate(atoms_fcc)
    assert calc.results["energy"] == pytest.approx(-13.71119226623514)
    assert np.sum(calc.results["forces"]) == pytest.approx(0.0)
    assert calc.delta_energy_10 == pytest.approx(0.0)
    assert calc.delta_energy_1a == pytest.approx(0.0)


def testPolymlpGeneralRefASECalculator(unitcell_mlp_Al):
    """Test PolymlpGeneralRefASECalculator."""
    unitcell, pot = unitcell_mlp_Al
    calc = PolymlpGeneralRefASECalculator(
        fc2,
        unitcell,
        pot_final=pot,
        pot_ref=pot,
        alpha=0.5,
    )
    assert calc._use_reference
    assert calc._use_fc2
    assert calc.alpha == 0.5

    calc.calculate(atoms_fcc)
    assert calc.results["energy"] == pytest.approx(-13.71119226623514)
    assert np.sum(calc.results["forces"]) == pytest.approx(0.0)
    assert calc.delta_energy_10 == pytest.approx(0.0)
    assert calc.delta_energy_1a == pytest.approx(0.0)
    assert calc.average_displacement == pytest.approx(0.0, abs=1e-8)
