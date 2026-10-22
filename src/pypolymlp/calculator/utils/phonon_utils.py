"""Utility functions for phonon calculations."""

import numpy as np
from phonopy import Phonopy

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)
from pypolymlp.utils.yaml_utils import load_cells


def load_phonon(
    yamlfile: str = "polymlp_phonon.yaml",
    filefc2: str = "fc2.hdf5",
    return_matrix: bool = True,
    return_phonopy: bool = False,
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

    if not return_phonopy:
        fc2 = load_fc2_hdf5(filefc2=filefc2, return_matrix=return_matrix)
        return (unitcell, supercell, fc2)

    fc2 = load_fc2_hdf5(filefc2=filefc2, return_matrix=False)
    unitcell_ph = structure_to_phonopy_cell(unitcell)
    ph = Phonopy(unitcell_ph, supercell.supercell_matrix, primitive_matrix="P")
    ph.force_constants = fc2
    return (unitcell, supercell, ph)


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


def produce_harmonic_force_constants(
    ph: Phonopy,
    prop: Properties,
    displacements: float = 0.01,
):
    """Produce non-effective harmonic FCs."""
    ph.generate_displacements(distance=displacements)
    supercells = ph.supercells_with_displacements
    supercells_pmlp = [phonopy_cell_to_structure(cell) for cell in supercells]
    _, forces, _ = prop.eval_multiple(supercells_pmlp)

    ph.forces = np.array(forces).transpose((0, 2, 1))
    ph.produce_force_constants()
    return ph.force_constants
