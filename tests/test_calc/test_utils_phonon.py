"""Tests of utilities for phonon calculations."""

from pathlib import Path

from pypolymlp.calculator.utils.phonon_utils import is_imaginary

cwd = Path(__file__).parent


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
