"""Tests of TiAl structure."""

from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "/mlps/polymlp.lammps.gtinv.Ti-Al"
polymlp = PypolymlpCalc(pot=pot)


def test_eval1():
    polymlp.load_poscars(path_file + "poscars/POSCAR-Al1.Ti-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-6.650991070412208, rel=1e-12)

    polymlp.load_poscars(path_file + "poscars/POSCAR-Al2.Ti-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-6.650991070412208, rel=1e-12)

    polymlp.load_poscars(path_file + "poscars/POSCAR-Ti1.Ti-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-10.413251553867987, rel=1e-12)

    polymlp.load_poscars(path_file + "poscars/POSCAR-Ti2.Ti-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-10.413251553867987, rel=1e-12)

    polymlp.load_poscars(path_file + "poscars/POSCAR-TiAl.Ti-Al")
    energies, _, _ = polymlp.eval()
    assert energies[0] == pytest.approx(-9.211925132828386, rel=1e-12)
