"""Tests of utilities for phonon calculations."""

from pathlib import Path

from pypolymlp.calculator.utils.phonon_utils import is_imaginary, load_phonon

cwd = Path(__file__).parent

path_files = str(cwd) + "/files/others/"


def test_is_imaginary():
    """Test is_imaginary."""
    freq = [0, 0.1, 0.2]
    dos = [0, 0.3, 0.5]
    assert not is_imaginary(freq, dos)

    freq = [-0.001, 0.1, 0.2]
    dos = [1e-3, 0.3, 0.5]
    assert not is_imaginary(freq, dos)

    freq = [-0.1, 0.1, 0.2]
    dos = [0.1, 0.3, 0.5]
    assert is_imaginary(freq, dos)


def test_load_phonon():
    """Test load phonon."""
    unitcell, supercell, fc2 = load_phonon(
        path_files + "polymlp_phonon_Ti.yaml",
        path_files + "fc2_Ti_222.hdf5",
    )
    assert len(unitcell.elements) == 2
    assert len(supercell.elements) == 16
    assert fc2.shape == (48, 48)
