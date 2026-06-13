"""Tests of C++ classes and functions."""

from pathlib import Path

import pytest

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
    assert len(ylm) == 66
    assert len(ylm_dx) == 66
    assert len(ylm_dy) == 66
    assert len(ylm_dz) == 66

    assert sum(ylm) == pytest.approx(
        -3.094632553138235 + 0.09510814961092404j, rel=1e-6
    )
    assert sum(ylm_dx) == pytest.approx(
        -6.916830463136405 - 1.910490992920798j, rel=1e-6
    )
    assert sum(ylm_dy) == pytest.approx(
        12.686387327323297 + 23.645024627283863j, rel=1e-6
    )
    assert sum(ylm_dz) == pytest.approx(
        -5.090360117438893 - 11.661266919165213j, rel=1e-6
    )
