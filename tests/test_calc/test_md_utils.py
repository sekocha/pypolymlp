"""Tests of utility functions for MD."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.md.md_utils import calc_integral, find_reference, get_p_roots

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_get_p_roots():
    """Test get_p_roots."""
    x, w = get_p_roots(n=5, a=0, b=1)
    np.testing.assert_allclose(x, [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
    np.testing.assert_allclose(
        w, [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]
    )


def test_calc_integral():
    """Test calc_integral using Gauss-Legendre quadrature."""
    x, w = get_p_roots(n=5, a=0.0, b=1.0)
    f = x**2
    val = calc_integral(w, f, a=0.0, b=1.0)
    assert val == pytest.approx(1 / 3)


def test_find_reference():
    """Test find_reference."""
    fc2file = find_reference(path_file + "others", 1000)
    assert fc2file.split("/")[-1] == "fc2.hdf5"
