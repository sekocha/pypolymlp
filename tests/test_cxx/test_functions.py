"""Tests of C++ classes and functions."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.cxx.api_functions import get_fn, get_ylm

cwd = Path(__file__).parent
path_files = str(cwd) + "/files/"


def test_get_fn():
    """Test for get_fn."""
    params = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    fn, fn_d = get_fn(r=1.2, params=params, cutoff=6.0)
    assert len(fn) == 3
    assert len(fn_d) == 3
    np.testing.assert_allclose(
        fn, [0.21430317094756238, 0.8690422117212636, 0.4769404780495180]
    )
    np.testing.assert_allclose(
        fn_d, [-0.5507864848009192, -0.495464911460655, 0.6819640474131319]
    )


def test_get_ylm():
    """Test for get_ylm."""
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
