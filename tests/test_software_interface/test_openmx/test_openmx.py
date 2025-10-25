"""Tests of openmx parser."""

from pathlib import Path

from pypolymlp.core.interface_openmx import parse_openmx

cwd = Path(__file__).parent


def test_load_openmx():
    """Test for loading openmx result files."""
    datafiles = [cwd / "AgC_444_5C_1.md"]
    structures, energies, forces = parse_openmx(datafiles)
    assert len(energies) == 207
    assert len(forces) == 207
    assert forces[0].shape == (3, 68)

    datafiles = [cwd / "AgC_444_5C_1.md.temp"]
    structures, energies, forces = parse_openmx(datafiles)
    assert len(energies) == 19
    assert len(forces) == 19
    assert forces[0].shape == (3, 68)
