"""Test normalize_positions."""

import itertools

import numpy as np

from pypolymlp.calculator.utils.structure_matcher.matcher import normalize_positions


def _test_normalize_positions(positions: np.ndarray, n_atoms: np.ndarray):
    """Test normalize_positions for a single set of fractional coordinates."""
    rep_true = normalize_positions(positions, n_atoms, decimals=5)
    trans_trials = np.random.rand(5, 3)
    for trans in trans_trials:
        trans = np.tile(trans, (positions.shape[1], 1)).T
        iperm = 0
        for p in itertools.permutations(range(positions.shape[1])):
            rep = normalize_positions(positions[:, p] + trans, n_atoms, decimals=5)
            np.testing.assert_allclose(rep, rep_true, atol=1e-5)
            iperm += 1
            if iperm > 10:
                break
        np.testing.assert_allclose(rep, rep_true, atol=1e-5)


def test_normalize_positions():
    """Test normalize_positions for a set of fractional coordinates."""
    n_atoms = [3]
    positions = np.array(
        [
            [0.0, 0.25, 0.1],
            [0.2, 0.25, 0.2],
            [0.6, 0.7, 0.9],
        ]
    ).T
    _test_normalize_positions(positions, n_atoms)


def test_normalize_positions2():
    """Test normalize_positions for a set of fractional coordinates."""
    n_atoms = [16]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.5, 0.0],
            [0.75, 0.5, 0.0],
            [0.46, 0.0, 0.25],
            [0.96, 0.0, 0.25],
            [0.21, 0.5, 0.25],
            [0.71, 0.5, 0.25],
            [0.42, 0.0, 0.5],
            [0.92, 0.0, 0.5],
            [0.17, 0.5, 0.5],
            [0.67, 0.5, 0.5],
            [0.21, 0.0, 0.75],
            [0.71, 0.0, 0.75],
            [0.46, 0.5, 0.75],
            [0.96, 0.5, 0.75],
        ]
    ).T
    _test_normalize_positions(positions, n_atoms)


def test_normalize_positions3():
    """Test normalize_positions for a set of fractional coordinates."""
    n_atoms = [8, 8]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.75, 0.5, 0.0],
            [0.46, 0.0, 0.25],
            [0.71, 0.5, 0.25],
            [0.42, 0.0, 0.5],
            [0.67, 0.5, 0.5],
            [0.21, 0.0, 0.75],
            [0.96, 0.5, 0.75],
            [0.5, 0.0, 0.0],
            [0.25, 0.5, 0.0],
            [0.96, 0.0, 0.25],
            [0.21, 0.5, 0.25],
            [0.92, 0.0, 0.5],
            [0.17, 0.5, 0.5],
            [0.71, 0.0, 0.75],
            [0.46, 0.5, 0.75],
        ]
    ).T
    _test_normalize_positions(positions, n_atoms)
