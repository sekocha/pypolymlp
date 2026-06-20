"""Test invariant list."""

import pytest

from pypolymlp.cxx.wrapper.api_gtinv_list import get_gtinv_attrs


def test_get_gtinv_attrs1():
    """Test invariant list."""
    l_combs, lm_seq, lm_coeffs = get_gtinv_attrs(
        order=3,
        lmax=[12, 12],
        version=1,
    )
    assert len(l_combs) == 154
    assert len(lm_seq) == 154
    assert len(lm_coeffs) == 154
    assert tuple(l_combs[23]) == (0, 9, 9)
    assert tuple(l_combs[145]) == (8, 12, 12)
    assert tuple(lm_seq[45][2]) == (4, 51, 97)
    assert lm_coeffs[45][2] == pytest.approx(-0.176928656723561)
    assert lm_coeffs[90][2] == pytest.approx(0.113929892405108)


def test_get_gtinv_attrs2():
    """Test invariant list."""
    l_combs, lm_seq, lm_coeffs = get_gtinv_attrs(
        order=6,
        lmax=[16, 12, 8, 1, 1],
        version=1,
    )
    assert len(l_combs) == 254
    assert len(lm_seq) == 254
    assert len(lm_coeffs) == 254
    assert tuple(l_combs[23]) == (0, 5, 5)
    assert tuple(l_combs[145]) == (8, 8, 12)
    assert tuple(lm_seq[45][2]) == (4, 27, 61)
    assert lm_coeffs[45][2] == pytest.approx(-0.18156825980064)


def test_get_gtinv_attrs3():
    """Test invariant list."""
    l_combs, lm_seq, lm_coeffs = get_gtinv_attrs(
        order=6,
        lmax=[30, 20, 10, 2, 2],
        version=2,
    )
    assert len(l_combs) == 1696
    assert len(lm_seq) == 1696
    assert len(lm_coeffs) == 1696
    assert tuple(l_combs[255]) == (6, 13, 19)
    assert tuple(l_combs[500]) == (14, 15, 19)
    assert tuple(lm_seq[1200][2]) == (16, 36, 46, 78)
    assert lm_coeffs[1300][2] == pytest.approx(0.0022483209653573586)
    assert lm_coeffs[1500][2] == pytest.approx(0.019665169858987183)
