"""Tests of C++ classes and functions."""

from pathlib import Path

from pypolymlp.cxx.api_cxx import get_ylm

cwd = Path(__file__).parent
path_files = str(cwd) + "/files/"


def test_get_ylm():
    """Test for loading polymlp files."""
    ylm, ylm_dx, ylm_dy, ylm_dz = get_ylm(
        r=1.0,
        polar=0.5,
        azimuthal=1.2,
        lmax=10,
    )
