"""Tests of neighbor calculations."""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent


def test_eos1():
    poscar = str(cwd) + "/POSCAR"
    pot = cwd / "polymlp.lammps"
    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.load_poscars(poscar)
    polymlp.run_eos(
        eps_min=0.7,
        eps_max=2.0,
        eps_step=0.03,
        fine_grid=True,
        eos_fit=True,
    )
    e0, v0, b0 = polymlp.eos_fit_data
    assert e0 == pytest.approx(-10.439420484403316, rel=1e-6)
    assert v0 == pytest.approx(34.37867473771466, rel=1e-6)
    assert b0 == pytest.approx(112.91376592916507, rel=1e-4)
