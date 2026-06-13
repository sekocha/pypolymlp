"""Tests of API for enumerating polynomial invariants."""

import os

import numpy as np
import pytest

from pypolymlp.polyinv.api_polyinv import (
    run_enum,
    run_enum_single_order,
    save_coeffs,
    save_coeffs_multiple_l,
    save_l,
    solve,
)


def test_run_enum():
    """Test for run_enum."""
    eigvecs_all, lm_indices_all = run_enum(
        orders=[2, 3, 4, 5, 6],
        maxl=1,
    )
    assert len(eigvecs_all) == 14
    assert len(lm_indices_all) == 14


def test_run_enum_single_order():
    """Test for run_enum_single_order."""
    eigvecs_all, lm_indices_all = run_enum_single_order(
        order=2,
        maxl=5,
        minl=None,
    )
    assert len(eigvecs_all) == 6
    assert len(lm_indices_all) == 6

    eigvecs_all, lm_indices_all = run_enum_single_order(
        order=3,
        maxl=5,
        minl=None,
    )
    assert len(eigvecs_all) == 20
    assert len(lm_indices_all) == 20


def test_solve_2():
    """Test for solve function."""
    eigvecs, lm_indices = solve([3, 3], lproj=0, verbose=True)
    assert eigvecs.shape == (7, 1)
    assert lm_indices.shape == (7, 2, 2)
    np.testing.assert_equal(lm_indices[2], [[3, -1], [3, 1]])

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.14285714285714335)

    eigvecs, lm_indices = solve([100, 100], lproj=0, verbose=True)
    assert eigvecs.shape == (201, 1)
    assert lm_indices.shape == (201, 2, 2)
    np.testing.assert_equal(lm_indices[2], [[100, -98], [100, 98]])

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.004975124378109379)


def test_solve_3():
    """Test for solve function."""
    eigvecs, lm_indices = solve([1, 2, 3], lproj=0, verbose=True)
    assert eigvecs.shape == (15, 1)
    assert lm_indices.shape == (15, 3, 2)
    lm_true = [[1, -1], [2, 0], [3, 1]]
    np.testing.assert_equal(lm_indices[2], lm_true)

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.08807358895321599)


def test_solve_4():
    """Test for solve function."""
    eigvecs, lm_indices = solve([1, 1, 2, 2], lproj=0, verbose=True)
    assert eigvecs.shape == (37, 3)
    assert lm_indices.shape == (37, 4, 2)
    lm_true = [[1, -1], [1, -1], [2, 2], [2, 0]]
    np.testing.assert_equal(lm_indices[2], lm_true)

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.09602028037268481)


def test_solve_5():
    """Test for solve function."""
    eigvecs, lm_indices = solve([1, 1, 1, 2, 3], lproj=0, verbose=True)
    assert eigvecs.shape == (125, 6)
    assert lm_indices.shape == (125, 5, 2)
    lm_true = [[1, -1], [1, -1], [1, -1], [2, 2], [3, 1]]
    np.testing.assert_equal(lm_indices[2], lm_true)

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.06452461399229097)


def test_solve_6():
    """Test for solve function."""
    eigvecs, lm_indices = solve([1, 1, 1, 1, 1, 1], lproj=0, verbose=True)
    assert eigvecs.shape == (141, 15)
    assert lm_indices.shape == (141, 6, 2)
    lm_true = [[1, -1], [1, -1], [1, 0], [1, 1], [1, 0], [1, 1]]
    np.testing.assert_equal(lm_indices[2], lm_true)

    proj = eigvecs @ eigvecs.T
    assert np.sum(proj) == pytest.approx(0.14285714285714524)


def test_save_coeffs():
    """Test save_coeffs."""
    eigvecs, lm_indices = solve([3, 3], lproj=0, verbose=True)
    save_coeffs(eigvecs, lm_indices, filename="tmp.yaml", mode="w", tag="inv")
    os.remove("tmp.yaml")


def test_save_coeffs_multiple_l():
    """Test save_coeffs_multiple_l."""
    eigvecs, lm_indices = solve([3, 3], lproj=0, verbose=True)
    save_coeffs_multiple_l([eigvecs], [lm_indices], filename="tmp.yaml")
    os.remove("tmp.yaml")


def test_save_l():
    """Test save_coeffs."""
    eigvecs, lm_indices = solve([3, 3], lproj=0, verbose=True)
    save_l([[0, 0], [0, 1]], [2, 1], filename="tmp.yaml")
    os.remove("tmp.yaml")
