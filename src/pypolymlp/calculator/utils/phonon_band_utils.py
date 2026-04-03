"""Utility functions for phonon band calculations."""

# import numpy as np
# from phonopy import Phonopy
#
from pypolymlp.calculator.utils.phonon_utils import load_phonon

# from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell
# from pypolymlp.utils.yaml_utils import load_cells


def calculate_phonon_bands(
    yamlfile: str = "polymlp_phonon.yaml",
    filefc2: str = "fc2.hdf5",
):
    """Calculate band structure from force constants and structure."""
    unitcell, supercell, phonopy = load_phonon(
        yamlfile=yamlfile,
        filefc2=filefc2,
        return_phonopy=True,
    )
