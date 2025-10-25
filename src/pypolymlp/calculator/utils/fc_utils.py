"""Utility functions for FC models."""

import h5py
import numpy as np


def load_fc2_hdf5(filefc2: str = "fc2.hdf5", return_matrix: bool = True):
    """Load FC2 in hdf5 format.

    Return
    ------
    fc2: Second-order force constants.
         shape=(N, N, 3, 3) or (N3, N3) if return_matrix == True.
    """
    h5file = h5py.File(filefc2, "r")
    fc = np.array(h5file["force_constants"])
    n_atom = fc.shape[1]
    if return_matrix:
        return fc.transpose((0, 2, 1, 3)).reshape((n_atom * 3, n_atom * 3))
    return fc


def recover_compact_fc2(fc2: np.ndarray):
    """Recover full FC2 from compact fc2."""
    pass


def eval_properties_fc2(fc2: np.ndarray, disps: np.ndarray):
    """Evaluate energy and forces from FC2.

    Parameters
    ----------
    fc2: Second-order force constants. shape=(N3, N3).
    disps: Displacements. shape=(N3).

    Return
    ------
    energy: Energy in eV.
    forces: Forces in eV/angstrom. shape=(3, N).
    """
    n_atom = fc2.shape[0] // 3
    disps_minus = -disps
    forces = fc2 @ disps_minus
    energy = disps_minus @ forces

    forces = forces.reshape((n_atom, 3)).T
    return 0.5 * energy, forces
