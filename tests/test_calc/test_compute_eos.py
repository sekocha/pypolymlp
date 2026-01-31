"""Tests of EOS calculation."""

import os
from pathlib import Path

import pytest

from pypolymlp.calculator.compute_eos import PolymlpEOS
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_eos_MgO():
    """Test EOS calculation."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    unitcell = Poscar(poscar).structure

    eos = PolymlpEOS(unitcell=unitcell, pot=pot)
    eos.run(eos_fit=True)

    assert eos.energy == pytest.approx(-40.391484997464076, rel=1e-6)
    assert eos.volume == pytest.approx(76.0684729812003, rel=1e-6)
    assert eos.bulk_modulus == pytest.approx(261.9462370383162, rel=1e-4)

    eos.write_eos_yaml(filename="tmp.yaml")
    os.remove("tmp.yaml")
