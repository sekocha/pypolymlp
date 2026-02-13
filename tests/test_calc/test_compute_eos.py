"""Tests of EOS calculation."""

import os

import pytest

from pypolymlp.calculator.compute_eos import PolymlpEOS


def _assert_eos_MgO(e0: float, v0: float, b0: float):
    """Assert EOS values."""
    assert e0 == pytest.approx(-40.391484997464076, rel=1e-6)
    assert v0 == pytest.approx(76.0684729812003, rel=1e-6)
    assert b0 == pytest.approx(261.9462370383162, rel=1e-4)


def test_eos_MgO(unitcell_pair_MgO):
    """Test EOS calculation."""
    unitcell, pot = unitcell_pair_MgO
    eos = PolymlpEOS(unitcell=unitcell, pot=pot)
    eos.run(eos_fit=True)
    _assert_eos_MgO(eos.energy, eos.volume, eos.bulk_modulus)

    eos.write_eos_yaml(filename="tmp.yaml")
    os.remove("tmp.yaml")
