"""Tests of harmonic reciprocal-part calculations."""

import os
import shutil

import numpy as np
import pytest

from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams


def test_harmonic_reciprocal(unitcell_mlp_Al):
    """Test HarmonicReciprocal."""
    unitcell, pot, prop = unitcell_mlp_Al
    size = (2, 2, 2)
    sscha_params = SSCHAParams(
        unitcell, size, pot=pot, temp=700, tol=0.003, symfc_use_mkl=False
    )
    sscha = SSCHACore(sscha_params, prop)
    rec = HarmonicReciprocal(sscha._phonopy, sscha._prop)

    fc2 = rec.produce_harmonic_force_constants()
    rec.force_constants = fc2
    rec.run(temp=700)

    assert fc2.shape == (32, 32, 3, 3)
    assert rec.force_constants.shape == (32, 32, 3, 3)

    assert rec.free_energy == pytest.approx(-63.73642039036109)
    assert rec.entropy == pytest.approx(192.44749588220637)
    assert rec.heat_capacity == pytest.approx(98.1658064023755)
    assert np.sum(rec.frequencies) == pytest.approx(2571.1278302298824)
    assert rec.phonopy_instance is not None

    f_true = -63.73642039036109
    f1, f2 = rec._compute_thermal_properties(temp=700, hide_imaginary=True)
    assert f1 == pytest.approx(f_true)
    assert f2 == pytest.approx(f_true)

    freq_true = 1.4693383503562443
    assert rec.frequencies[0, 2] == pytest.approx(freq_true)
    assert rec.mesh_dict["frequencies"][0, 2] == pytest.approx(freq_true)
    assert rec.tp_dict["free_energy"][0] == pytest.approx(f_true)
    assert not rec.is_imaginary

    rec._print_frequencies()

    rec.calc_save_dos(filename="tmp.dat", qmesh=(2, 2, 2))
    os.remove("tmp.dat")

    rec.calc_save_dos(filename="tmp/total_dos.dat", qmesh=(2, 2, 2), save_pdos=True)
    shutil.rmtree("tmp")
