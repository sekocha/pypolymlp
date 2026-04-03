"""Tests of utilities for phonon band calculations."""

import os
from pathlib import Path

from pypolymlp.calculator.utils.phonon_band_utils import calculate_phonon_bands

cwd = Path(__file__).parent

path_files = str(cwd) + "/files/others/"


def test_calculate_phonon_bands():
    """Test calculate_phonon_bands."""
    calculate_phonon_bands(
        yamlfile=path_files + "polymlp_phonon_Ti.yaml",
        filefc2=path_files + "fc2_Ti_222.hdf5",
        structure_type="perovskite",
    )
    os.remove("phonon_band.yaml")
    os.remove("phonon_band.dat")
