"""Tests of utility functions for force constant calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.utils.fc_utils import eval_properties_fc2, load_fc2_hdf5

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_fc2_functions():
    """Test load_fc2_hdf5 and eval_properties_fc2."""
    fc2hdf5 = path_file + "others/sscha_restart/fc2.hdf5"
    fc2 = load_fc2_hdf5(fc2hdf5, return_matrix=False)
    assert fc2.shape == (32, 32, 3, 3)

    fc2 = load_fc2_hdf5(fc2hdf5)
    assert fc2.shape == (96, 96)

    # Test eval_properties_fc2
    disps = np.zeros((96))
    disps[3] = 0.002
    disps[10] = 0.001
    disps[20] = -0.001
    energy, forces = eval_properties_fc2(fc2, disps)
    assert energy == pytest.approx(1.5667425536505865e-05)
    assert forces.shape == (3, 32)
