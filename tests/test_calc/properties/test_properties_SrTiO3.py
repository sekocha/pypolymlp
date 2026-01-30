"""Tests of neighbor calculations."""

import glob
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

unitcell = Poscar(path_file + "poscars/POSCAR.perovskite.SrTiO3").structure


def test_eval_cond():
    """Test polymlp with conditional conditions of pair distances."""
    pot = path_file + "/mlps/polymlp.yaml.gtinv.cond.SrTiO3"
    prop = Properties(pot=pot)

    energy_true = -31.642024569145274
    stresses_true = [0.21674893, 0.21674893, 0.21674893, 0, 0, 0]
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-8)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval_flexible():
    """Test polymlp with flexible selection of pair features."""
    files = sorted(glob.glob(path_file + "mlps/polymlp.yaml.flexible.*"))
    prop = Properties(pot=files)

    energy_true = -31.642657299368572
    energy, forces, stresses = prop.eval(unitcell)
    assert energy == pytest.approx(energy_true, rel=1e-6)
    assert forces[0][0] == pytest.approx(0.0, abs=1e-12)

    # stresses_true = [0.05689321, 0.05689321, 0.05689321, 0, 0, 0]
    # stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    # np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
