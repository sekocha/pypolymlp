"""Tests of functions and attributes for property calculations."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import (
    Properties,
    convert_stresses_in_gpa,
    find_active_atoms,
)
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

unitcell = Poscar(path_file + "poscars/POSCAR.RS.MgO").structure


def test_eval_single():
    """Test properties with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    prop = Properties(pot=pot)
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-40.22469744308482)

    energies, forces, stresses = prop.eval_multiple([unitcell, unitcell])
    np.testing.assert_allclose(energies, [-40.22469744308482] * 2)
    np.testing.assert_allclose(prop.energies, [-40.22469744308482] * 2)
    assert np.array(prop.forces).shape == (2, 3, 8)
    assert prop.stresses.shape == (2, 6)

    prop.save()
    prop.print_single()
    os.remove("polymlp_energies.npy")
    os.remove("polymlp_stress_tensors.npy")
    os.remove("polymlp_forces.npy")

    assert prop.params.n_type == 2


def test_eval_hybrid():
    """Test properties with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    prop = Properties(pot=[pot, pot])
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(-40.22469744308482 * 2)

    energies, forces, stresses = prop.eval_multiple([unitcell, unitcell])
    np.testing.assert_allclose(energies, [-40.22469744308482 * 2] * 2)
    np.testing.assert_allclose(prop.energies, [-40.22469744308482 * 2] * 2)
    assert np.array(prop.forces).shape == (2, 3, 8)
    assert prop.stresses.shape == (2, 6)
    assert prop.stresses_gpa.shape == (2, 6)


def test_find_active_atoms():
    """Test find_active_atoms."""
    element_order = ["Mg"]
    structures = [unitcell, unitcell]
    strs, active_atoms, active_bools = find_active_atoms(structures, element_order)
    assert len(strs[0].elements) == 4
    assert len(strs[1].elements) == 4
    np.testing.assert_equal(active_atoms[0], [0, 1, 2, 3])
    np.testing.assert_equal(active_atoms[1], [0, 1, 2, 3])
    np.testing.assert_equal(active_bools, True)

    element_order = ["O"]
    structures = [unitcell, unitcell]
    strs, active_atoms, active_bools = find_active_atoms(structures, element_order)
    assert len(strs[0].elements) == 4
    assert len(strs[1].elements) == 4
    np.testing.assert_equal(active_atoms[0], [4, 5, 6, 7])
    np.testing.assert_equal(active_atoms[1], [4, 5, 6, 7])
    np.testing.assert_equal(active_bools, True)

    element_order = ["S"]
    structures = [unitcell, unitcell]
    strs, active_atoms, active_bools = find_active_atoms(structures, element_order)
    assert len(strs) == 0
    assert len(active_atoms) == 0
    np.testing.assert_equal(active_bools, False)


def test_convert_stresses_in_gpa():
    """Test convert_stresses_in_gpa."""
    stresses = np.array(
        [[2.0, 3.0, 4.0, -1.0, 0.5, 0.2], [2.0, 3.0, 4.0, -1.0, 0.5, 0.2]]
    )
    stresses_gpa = convert_stresses_in_gpa(stresses, [unitcell, unitcell])
    true = [4.18769325, 6.28153988, 8.37538651, -2.09384663, 1.04692331, 0.41876933]
    np.testing.assert_allclose(stresses_gpa[0], true)
