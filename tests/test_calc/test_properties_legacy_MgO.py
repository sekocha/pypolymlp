"""Tests of property calculations using legacy polymlp in MgO."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

unitcell = Poscar(path_file + "poscars/POSCAR.RS.MgO").structure


def test_eval1():
    """Tests property calculations using legacy polymlp in MgO."""
    pot = path_file + "mlps/polymlp.lammps.pair.MgO"
    prop = Properties(pot=pot)
    energy, forces, stresses = prop.eval(unitcell)

    energy_true = -40.22469744315832
    forces_true = [
        [-0.03962958, -0.01188776, -0.07928375],
        [-0.00459307, 0.00331611, 0.02210267],
        [0.01105381, -0.00137797, 0.022104],
        [0.01105096, 0.00331545, -0.00918516],
        [0.00978188, 0.00177061, 0.01180386],
        [0.00590284, 0.00293358, 0.01180553],
        [0.00589926, 0.00176979, 0.01958533],
        [0.0005339, 0.00016018, 0.00106752],
    ]
    stresses_true = [
        -0.17379186,
        -0.17521681,
        -0.16909401,
        0.00004465,
        0.00008926,
        0.00029751,
    ]

    assert energy == pytest.approx(energy_true, rel=1e-8)
    np.testing.assert_allclose(forces.T, forces_true, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [unitcell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
