"""Tests of GSFE calculation."""

import copy
import os
import shutil

import pytest

from pypolymlp.calculator.compute_gsfe import PolymlpGSFE


def test_gsfe1(unitcell_mlp_Al):
    """Test GSFE calculation."""
    unitcell1, pot, prop = unitcell_mlp_Al
    unitcell = copy.deepcopy(unitcell1)
    trans = PolymlpGSFE(structure=unitcell, properties=prop, verbose=True)
    trans.set_supercell(
        disp1=(1, 0, 0),
        disp2=(0, 1, -1),
        slip_plane=(0, 1, 1),
        n_layers=2,
    )
    trans.run(n_points=2)
    trans.save(filename="tmp.dat")
    os.remove("tmp.dat")
    shutil.rmtree("poscars")
    assert trans.excess_energies.shape == (9, 3)
    assert trans.excess_energies[5, 2] == pytest.approx(0.5883636566785202)
    assert trans.excess_energies[-1, 2] == pytest.approx(0.636360235)
