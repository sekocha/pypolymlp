"""Tests of harmonic reciprocal-part calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"

unitcell = Poscar(poscar).structure
size = (2, 2, 2)


def test_harmonic_reciprocal():
    """Test HarmonicReciprocal."""
    sscha_params = SSCHAParams(unitcell, size, pot=pot, temp=700, tol=0.003)
    sscha = SSCHACore(sscha_params, pot=pot)
    rec = HarmonicReciprocal(sscha._phonopy, sscha._prop)
    energies, forces = rec.eval([unitcell, unitcell])
    np.testing.assert_allclose(energies, -13.71119227, atol=1e-7)
    np.testing.assert_allclose(forces, 0.0, atol=1e-7)

    fc2 = rec.produce_harmonic_force_constants()
    rec.compute_thermal_properties(temp=700)
    rec.compute_mesh_properties()
    assert fc2.shape == (32, 32, 3, 3)
    assert rec.force_constants.shape == (32, 32, 3, 3)

    assert rec.free_energy == pytest.approx(-63.73642039036109)
    assert rec.entropy == pytest.approx(192.44749588220637)
    assert rec.heat_capacity == pytest.approx(98.1658064023755)
    assert np.sum(rec.frequencies) == pytest.approx(2571.1278302298824)
    assert rec.phonopy_instance is not None
