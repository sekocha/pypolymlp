"""Tests of thermodynamics_transition.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.thermodynamics.transition import (
    compute_phase_boundary,
    find_transition,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_find_transition():
    """Test find_transition."""
    f1 = np.array([[100, 0.5], [200, 0.25], [300, 0], [400, -0.2], [500, -0.4]])
    f2 = np.array([[120, -0.5], [220, -0.1], [320, 0.1], [420, 0.15], [520, 0.3]])
    tc_fit = find_transition(f1, f2)
    assert tc_fit == pytest.approx(286.666, rel=1e-5)


def test_compute_phase_boundary():
    """Test compute_phase_boundary."""
    g1 = np.array(
        [
            [[-5, 1.5], [-3, 1.25], [-1, 1.0], [1, -1.2], [4, -1.4]],
            [[-5, 1.4], [-3, 1.15], [-1, 1.1], [1, -1.1], [4, -1.3]],
            [[-5, 1.3], [-3, 1.05], [-1, 1.2], [1, -1], [4, -1.2]],
        ]
    )
    g2 = np.array(
        [
            [[-4.5, 0.5], [-2.5, 0.25], [0, 0.0], [2.0, -0.2], [3.0, -0.4]],
            [[-4.5, 0.3], [-2.5, 0.05], [0, 0.1], [2.0, -0.3], [3.0, -0.5]],
            [[-4.5, 0.1], [-2.5, 0.05], [0, 0.2], [2.0, -0.4], [3.0, -0.6]],
        ]
    )
    temperatures = [300, 500, 700]
    boundary = compute_phase_boundary(
        g1,
        temperatures,
        g2,
        temperatures,
        fit_gibbs_max_order=3,
    )
    true = [
        [0.0, 272.65282677],
        [0.25, 410.14354854],
        [0.5, 598.17756491],
        [0.75, 819.54830784],
        [1.0, 929.85826623],
        [1.25, 986.61350865],
        [1.5, 1014.75289422],
        [1.75, 1026.58376299],
        [2.0, 1028.73471756],
        [2.25, 1025.00570316],
    ]
    np.testing.assert_allclose(boundary, true)
