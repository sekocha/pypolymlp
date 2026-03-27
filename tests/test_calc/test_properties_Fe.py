"""Tests of property calculations in Fe."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.data_format import PolymlpStructure

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


axis = np.eye(3) * 4.0
positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
).T
n_atoms = [2, 2]
elements = ["Fe", "Fe", "Fe", "Fe"]
types = [0, 0, 1, 1]
cell = PolymlpStructure(
    axis=axis,
    positions=positions,
    n_atoms=n_atoms,
    elements=elements,
    types=types,
)


def test_eval1():
    """Test properties with polymlp in Fe with spin configurations."""
    pot = path_file + "mlps/polymlp.yaml.1.Fe"
    prop = Properties(pot=pot)

    energy, forces, stresses = prop.eval(cell)

    energy_true = -18.04016217212223
    stresses_true = [
        -58.01061818705562,
        -63.75795779528501,
        -63.75795779528501,
        0.0,
        0.0,
        0.0,
    ]

    assert energy == pytest.approx(energy_true, rel=1e-6)
    np.testing.assert_allclose(forces, 0.0, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [cell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)


def test_eval2():
    """Test properties with polymlp in Fe with spin configurations."""
    pot = [
        path_file + "mlps/polymlp.yaml.1.Fe",
        path_file + "mlps/polymlp.yaml.2.Fe",
    ]
    prop = Properties(pot=pot)

    energy, forces, stresses = prop.eval(cell)

    energy_true = -16.829485851435084
    stresses_true = [-10.21205091, -24.54044123, -24.54044123, 0.0, 0.0, 0.0]

    assert energy == pytest.approx(energy_true, rel=1e-6)
    np.testing.assert_allclose(forces, 0.0, atol=1e-6)
    stresses = convert_stresses_in_gpa(np.array([stresses]), [cell])[0]
    np.testing.assert_allclose(stresses, stresses_true, atol=1e-5)
