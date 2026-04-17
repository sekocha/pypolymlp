"""Utility functions for SSCHA."""

import numpy as np


def symmetrize_properties(
    forces: np.ndarray,
    stress: np.ndarray,
    proj_forces: np.ndarray,
    proj_stress: np.ndarray,
    n_unitcells: int,
):
    """SymmeWrite SSCHA results to a file."""
    n_atom_supercell = forces.shape[1]

    forces_sym = (proj_forces @ forces.T.reshape(-1)).reshape((-1, 3)).T
    unitcell_reps = np.arange(n_atom_supercell) % n_unitcells == 0
    forces_sym = forces_sym[:, unitcell_reps]

    order = [0, 3, 5, 3, 1, 4, 5, 4, 2]
    stress_sym = proj_stress @ stress[order]
    order = [0, 4, 8, 1, 5, 6]
    stress_sym = stress_sym[order]
    return forces_sym, stress_sym
