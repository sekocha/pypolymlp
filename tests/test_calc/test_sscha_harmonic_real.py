"""Tests of harmonic real-part calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"

unitcell = Poscar(poscar).structure
size = (2, 2, 2)


def test_harmonic_real():
    """Test HarmonicReal."""
    sscha_params = SSCHAParams(unitcell, size, pot=pot, temp=700, tol=0.003)
    sscha = SSCHACore(sscha_params, pot=pot)

    phonopy = sscha._phonopy
    supercell_polymlp = phonopy_cell_to_structure(phonopy.supercell)
    supercell_polymlp.masses = phonopy.supercell.masses
    supercell_polymlp.supercell_matrix = sscha_params.supercell_matrix
    supercell_polymlp.n_unitcells = sscha_params.n_unitcells
    sscha_params.supercell = supercell_polymlp

    path_sscha = path_file + "others/sscha_restart/"
    fc2hdf5 = path_sscha + "fc2.hdf5"
    fc2 = load_fc2_hdf5(fc2hdf5, return_matrix=False)
    real = HarmonicReal(supercell_polymlp, sscha._prop, fc2=fc2)
    energies, forces = real.eval([unitcell, unitcell])
    np.testing.assert_allclose(energies, -13.71119227, atol=1e-7)
    np.testing.assert_allclose(forces, 0.0, atol=1e-7)

    real.run(temp=700, n_samples=10)

    assert real.force_constants.shape == (32, 32, 3, 3)
    real.force_constants = real.force_constants

    assert real.displacements.shape == (10, 3, 32)
    assert real.supercells is not None
    assert real.forces.shape == (10, 3, 32)
    assert real.full_potentials.shape == (10,)
    assert real.average_full_potential is not None
    assert real.harmonic_potentials.shape == (10,)
    assert real.average_harmonic_potential is not None
    assert real.anharmonic_potentials.shape == (10,)
    assert real.average_anharmonic_potential is not None
    assert real.static_potential == pytest.approx(-1322.92893961425)
    assert np.sum(real.static_forces) == pytest.approx(0.0)
    assert np.sum(real.average_forces) == pytest.approx(0.0)
    assert real.frequencies.shape == (96,)
