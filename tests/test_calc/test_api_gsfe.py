"""Tests of GSFEs using API."""

import copy
import os
import shutil
from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_gsfe1(unitcell_mlp_Al):
    """Test GSFE calculation."""
    unitcell1, pot, _ = unitcell_mlp_Al

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.structures = copy.deepcopy(unitcell1)
    excess_energies = polymlp.run_gsfe(
        disp1=(1, 0, 0),
        disp2=(0, 1, -1),
        glide_plane=(0, 1, 1),
        n_layers=2,
        n_points=2,
        filename="tmp.dat",
    )
    os.remove("tmp.dat")
    shutil.rmtree("poscars")
    assert excess_energies.shape == (9, 3)
    assert excess_energies[5, 2] == pytest.approx(0.5883636566785202)
    assert excess_energies[-1, 2] == pytest.approx(0.636360235)


def test_gsfe_single(unitcell_mlp_Al):
    """Test GSFE single-point calculation."""
    unitcell1, pot, _ = unitcell_mlp_Al

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.structures = copy.deepcopy(unitcell1)
    energy = polymlp.run_gsfe(
        disp1=(1, 0, 0),
        disp2=(0, 1, -1),
        glide_plane=(0, 1, 1),
        n_layers=2,
        frac1=0.25,
        frac2=0.3,
        filename="tmp.dat",
    )
    assert energy == pytest.approx(-18.19393440569258)
    os.remove("tmp.dat")
    os.remove("POSCAR_gsf")
