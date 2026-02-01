"""Tests of force constant calculations."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_fc_AlN():
    """Test FC calculation."""
    poscar = path_file + "poscars/POSCAR.WZ.AlN"
    pot = path_file + "mlps/polymlp.lammps.gtinv.AlN"

    unitcell = Poscar(poscar).structure
    supercell = supercell_diagonal(unitcell, size=(3, 3, 2), use_phonopy=True)
    fc = PolymlpFC(
        supercell=supercell,
        pot=pot,
        cutoff=3.0,
        verbose=True,
    )
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


def test_fc_MgO():
    """Test FC calculation."""
    poscar = path_file + "poscars/POSCAR.RS.IdealMgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"

    unitcell = Poscar(poscar).structure
    supercell = supercell_diagonal(unitcell, size=(2, 2, 2), use_phonopy=True)
    fc = PolymlpFC(
        supercell=supercell,
        pot=pot,
        cutoff=3.0,
        verbose=True,
    )
    fc.sample(n_samples=100, displacements=0.01)
    fc.run(orders=(2, 3), write_fc=False, use_mkl=False)

    assert fc.fc3.shape == (2, 64, 64, 3, 3, 3)
    assert np.sum(fc.fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc.fc3) == pytest.approx(0.0, abs=1e-6)
