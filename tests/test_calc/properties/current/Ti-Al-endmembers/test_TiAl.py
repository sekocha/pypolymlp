"""Tests of TiAl structure."""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent


def test_eval1():
    polymlp = PypolymlpCalc(pot=cwd / "mlp.lammps")
    polymlp.load_poscars(str(cwd) + "/POSCAR-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-6.650991070412208, rel=1e-12)

    polymlp.load_poscars(str(cwd) + "/POSCAR-Al2")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-6.650991070412208, rel=1e-12)

    polymlp.load_poscars(str(cwd) + "/POSCAR-Ti")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-10.413251553867987, rel=1e-12)

    polymlp.load_poscars(str(cwd) + "/POSCAR-Ti2")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-10.413251553867987, rel=1e-12)

    polymlp.load_poscars(str(cwd) + "/POSCAR-TiAl")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-9.211925132828386, rel=1e-12)
