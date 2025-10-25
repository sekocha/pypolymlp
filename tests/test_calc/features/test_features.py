"""Tests of neighbor calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent


def test_features1():
    poscars = [cwd / "poscar-00001", cwd / "poscar-00002"]
    pot = cwd / "polymlp.lammps"
    polymlp = PypolymlpCalc(pot=pot, verbose=True, require_mlp=True)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(features_force=False, features_stress=False)
    x = polymlp.features
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)

    infile = cwd / "polymlp.in"
    polymlp = PypolymlpCalc(verbose=True, require_mlp=False)
    polymlp.load_structures_from_files(poscars=poscars)
    polymlp.run_features(
        develop_infile=infile, features_force=False, features_stress=False
    )
    x = polymlp.features
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)
