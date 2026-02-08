"""Tests of attributes and functions of API for polynomial MLP calculation."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_files = str(cwd) + "/files/"


def test_load_mlp():
    """Test for loading polymlp files."""
    filename = path_files + "mlps/polymlp.lammps.pair.MgO"
    coeff_true = 9.352307613515078e00 / 2.067583465937491e-01

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop._prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    filename = path_files + "mlps/polymlp.yaml.gtinv.MgO"
    coeff_true = 5.794375827500248e01

    mlp_calc = PypolymlpCalc(pot=filename)
    coeffs = mlp_calc._prop._prop._coeffs
    assert coeffs[0] == pytest.approx(coeff_true, rel=1e-8)


def test_functions1():
    """Test functions in API."""
    filename = path_files + "mlps/polymlp.lammps.pair.MgO"
    calc = PypolymlpCalc(pot=filename)

    poscar1 = path_files + "poscars/POSCAR-00001.MgO"
    poscar2 = path_files + "poscars/POSCAR-00002.MgO"
    calc.load_poscars(poscar1)
    assert len(calc.structures) == 1
    calc.load_poscars([poscar1, poscar2])
    assert len(calc.structures) == 2

    vasprun1 = path_files + "others/vasprun-00001-MgO.xml"
    vasprun2 = path_files + "others/vasprun-00002-MgO.xml"
    calc.load_vaspruns(vasprun1)
    assert len(calc.structures) == 1
    calc.load_vaspruns([vasprun1, vasprun2])
    assert len(calc.structures) == 2

    calc.load_structures_from_files(poscars=[poscar1, poscar2])
    assert len(calc.structures) == 2
    calc.load_structures_from_files(vaspruns=[vasprun1, vasprun2])
    assert len(calc.structures) == 2

    calc.save_poscars(prefix="tmp-")
    os.remove("tmp-000")
    os.remove("tmp-001")

    energies, forces, stresses = calc.eval()
    assert energies.shape == (2,)
    assert np.array(forces).shape == (2, 3, 64)
    assert stresses.shape == (2, 6)

    calc.save_properties()
    os.remove("polymlp_energies.npy")
    os.remove("polymlp_forces.npy")
    os.remove("polymlp_stress_tensors.npy")
    calc.print_properties()

    assert calc.energies.shape == (2,)
    assert np.array(calc.forces).shape == (2, 3, 64)
    assert calc.stresses.shape == (2, 6)
    assert calc.stresses_gpa.shape == (2, 6)


def test_attrs_empty_calc():
    """Test attributes in API."""
    filename = path_files + "mlps/polymlp.lammps.pair.MgO"
    calc = PypolymlpCalc(pot=filename)

    assert calc.instance_properties is not None
    assert calc.params is not None
    assert calc.energies is None
    assert calc.forces is None
    assert calc.stresses is None
    assert calc.stresses_gpa is None
    assert calc.structures is None
    assert calc.first_structure is None
    assert calc.converged_structure is None
    assert calc.structures is None
    assert calc.unitcell is None
    assert calc.unitcell is None
    assert calc.features is None
    assert calc.elastic_constants is None
    assert calc.eos_fit_data is None
    assert calc.eos_curve_data is None
    assert calc.go_data is None
    assert calc.instance_phonopy is None
    assert calc.phonon_dos is None
    assert calc.is_imaginary is None
    assert calc.temperatures is None
    assert calc.bulk_modulus_temperature is None
    assert calc.thermal_expansion is None
