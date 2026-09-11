"""Tests of transformation pathway."""

import copy
import os
import shutil

import pytest

from pypolymlp.calculator.compute_transformation import PolymlpTransformation


def test_tr1(unitcell_mlp_Al):
    """Test transformation pathway calculation."""
    unitcell1, pot, prop = unitcell_mlp_Al
    unitcell = copy.deepcopy(unitcell1)

    trans = PolymlpTransformation(unitcell, prop, verbose=False)
    trans.set_supercell(
        disp1=(1, 0, 0),
        disp2=(0, 1, 0),
        slip_plane=(0, 0, 1),
        n_layers=3,
    )

    trans.run_fix_angle(
        degs_min=90, degs_max=95, degs_int=1, axis1=0, axis2=2, gtol=1e-4
    )
    trans.save(filename="tmp.dat")
    os.remove("tmp.dat")
    shutil.rmtree("poscars")
    assert trans.energies.shape == (6, 2)
    assert trans.energies[3, 1] == pytest.approx(0.004978147064782353)
    assert trans.energies[-1, 1] == pytest.approx(0.013258090659961752)


def test_tr2(unitcell_mlp_Al):
    """Test transformation pathway calculation."""
    unitcell1, pot, prop = unitcell_mlp_Al
    unitcell = copy.deepcopy(unitcell1)

    trans = PolymlpTransformation(unitcell, prop, verbose=False)
    trans.set_supercell(
        disp1=(1, 0, 0),
        disp2=(0, 1, 0),
        slip_plane=(0, 0, 1),
        n_layers=4,
    )
    trans.run_fix_shift(
        max_shift_frac=0.5,
        n_points=5,
        axis_shift=0,
        axis_normal_shift=2,
        gtol=1e-4,
    )
    trans.save(filename="tmp.dat")
    os.remove("tmp.dat")
    shutil.rmtree("poscars")

    assert trans.energies.shape == (6, 2)
    assert trans.energies[3, 1] == pytest.approx(0.04286155857041152)
    assert trans.energies[-1, 1] == pytest.approx(0.07796180414937437)
