"""Tests of phonon calculation."""

import os
import shutil

import numpy as np

from pypolymlp.calculator.compute_phonon import PolymlpPhonon, PolymlpPhononQHA


def _assert_phonon(ph: PolymlpPhonon):
    """Assert phonon calculations."""
    assert ph.total_dos.shape == (201, 2)
    assert not ph.is_imaginary

    keys = ph.thermal_properties.keys()
    assert "temperatures" in keys
    assert "free_energy" in keys
    assert "entropy" in keys
    assert "heat_capacity" in keys

    ph.write_properties()
    os.remove("phonon_mesh_qpoints.txt")
    os.remove("phonon_thermal_properties.yaml")
    os.remove("phonon_total_dos.dat")


def _assert_qha(ph: PolymlpPhononQHA):
    """Assert phonon calculations."""
    ph.write_qha()
    shutil.rmtree("polymlp_phonon_qha")
    assert len(ph.temperatures) == 101
    assert len(ph.bulk_modulus_temperature) == 100
    assert len(ph.thermal_expansion) == 100


def test_phonon_MgO(unitcell_pair_MgO):
    """Test EOS calculation."""
    unitcell, pot = unitcell_pair_MgO
    ph = PolymlpPhonon(unitcell=unitcell, supercell_matrix=np.diag([2, 2, 2]), pot=pot)
    ph.produce_force_constants(distance=0.01)
    ph.compute_properties()
    _assert_phonon(ph)


def test_phonon_qha_MgO(unitcell_pair_MgO):
    """Test EOS calculation."""
    unitcell, pot = unitcell_pair_MgO
    ph = PolymlpPhononQHA(unitcell=unitcell, supercell_matrix=np.eye(3), pot=pot)
    ph.run()
    _assert_qha(ph)
