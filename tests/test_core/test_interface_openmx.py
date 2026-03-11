"""Tests of openmx parser."""

from pathlib import Path

from pypolymlp.core.interface_openmx import parse_openmx, set_dataset_from_openmx

cwd = Path(__file__).parent


def test_set_dataset_from_openmx():
    """Test set_dataset_from_openmx."""
    dft = set_dataset_from_openmx(cwd / "./../files/openmx.AgC_444_5C_1.md")
    assert len(dft.energies) == 207
    assert len(dft.forces) == 42228
    dft = set_dataset_from_openmx([cwd / "./../files/openmx.AgC_444_5C_1.md"])
    assert len(dft.energies) == 207
    assert len(dft.forces) == 42228


def test_load_openmx():
    """Test for loading openmx result files."""
    datafiles = [cwd / "./../files/openmx.AgC_444_5C_1.md"]
    structures, energies, forces = parse_openmx(datafiles)
    assert len(energies) == 207
    assert len(forces) == 207
    assert forces[0].shape == (3, 68)

    datafiles = [cwd / "./../files/openmx.AgC_444_5C_1.md.temp"]
    structures, energies, forces = parse_openmx(datafiles)
    assert len(energies) == 19
    assert len(forces) == 19
    assert forces[0].shape == (3, 68)
