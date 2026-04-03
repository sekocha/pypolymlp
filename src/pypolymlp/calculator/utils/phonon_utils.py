"""Utility functions for phonon calculations."""

import numpy as np

from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.utils.yaml_utils import load_cells


def load_phonon(
    yamlfile: str = "polymlp_phonon.yaml",
    filefc2: str = "fc2.hdf5",
    return_matrix: bool = True,
):
    """Load unitcell, supercell, and FC2.

    Return
    ------
    unitcell: Unitcell in PolymlpStructure.
    supercell: Supercell in PolymlpStructure.
    fc2: Second-order force constants.
         shape=(N, N, 3, 3) or (N3, N3) if return_matrix == True.
    """
    unitcell, supercell = load_cells(filename=yamlfile)
    fc2 = load_fc2_hdf5(filefc2=filefc2, return_matrix=return_matrix)
    return (unitcell, supercell, fc2)


def is_imaginary(
    frequencies: np.ndarray,
    dos: np.ndarray,
    tol_frequency: float = -0.01,
):
    """Check if phonon DOS has imaginary frequencies."""
    frequencies = np.array(frequencies)
    dos = np.array(dos)
    is_imag = frequencies < tol_frequency
    return np.sum(dos[is_imag]) > 1e-6
