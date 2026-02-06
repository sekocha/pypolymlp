"""Tests of force constant calculations."""

import os

import numpy as np
import pytest

from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import supercell


def test_fc_AlN(unitcell_wz_AlN):
    """Test FC calculation."""
    unitcell, pot = unitcell_wz_AlN
    sup = supercell(unitcell, (3, 3, 2), use_phonopy=True)
    fc = PolymlpFC(supercell=sup, pot=pot, cutoff=3.0, verbose=True)
    fc.sample(n_samples=100, displacements=0.01)
    fc.run(orders=(2, 3), write_fc=False, use_mkl=False)

    assert fc.fc3.shape == (4, 72, 72, 3, 3, 3)
    assert np.sum(fc.fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc.fc3) == pytest.approx(0.0, abs=1e-6)

    fc.save_fc()
    os.remove("fc2.hdf5")
    os.remove("fc3.hdf5")

    assert fc.displacements.shape == (100, 3, 72)
    assert fc.forces.shape == (100, 3, 72)
    assert len(fc.structures) == 100
    fc.displacements = fc.displacements
    fc.forces = fc.forces
    assert fc.cutoff == pytest.approx(3.0)
    assert isinstance(fc.supercell, PolymlpStructure)


def test_fc_MgO(unitcell_pair_MgO):
    """Test FC calculation."""
    unitcell, pot = unitcell_pair_MgO
    sup = supercell(unitcell, (2, 2, 2), use_phonopy=True)
    fc = PolymlpFC(supercell=sup, pot=pot, cutoff=3.0, verbose=True)
    fc.sample(n_samples=100, displacements=0.01)
    fc.run(orders=(2, 3), write_fc=False, use_mkl=False)

    assert fc.fc3.shape == (2, 64, 64, 3, 3, 3)
    assert np.sum(fc.fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc.fc3) == pytest.approx(0.0, abs=1e-6)
