"""Tests of vasprun compress functions."""

from pathlib import Path

from pypolymlp.utils.vasprun_compress import compress_vaspruns

cwd = Path(__file__).parent


def test_compress_vaspruns():
    """Test for compress_vaspruns."""
    filename = str(cwd) + "/../files/vasprun-00001-Ti-full.xml"
    success = compress_vaspruns(filename, write_file=False)
    assert success
