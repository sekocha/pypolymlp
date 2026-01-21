"""Tests of units.py."""

from pypolymlp.core.units import (
    Avogadro,
    BohrtoAng,
    EVtoGPa,
    EVtoJ,
    EVtoJmol,
    EVtoKbar,
    EVtoKJmol,
    HartreetoEV,
    Kb,
    KbEV,
    M_StoAng_Fs,
    MasstoKG,
    Planck,
)


def test_units():
    """Test units."""
    _ = Avogadro
    _ = Planck
    _ = Kb
    _ = KbEV
    _ = EVtoJ
    _ = EVtoKJmol
    _ = EVtoJmol
    _ = EVtoKbar
    _ = EVtoGPa
    _ = MasstoKG
    _ = M_StoAng_Fs
    _ = HartreetoEV
    _ = BohrtoAng
