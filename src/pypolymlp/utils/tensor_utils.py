"""Utilities to calculate tensor properties."""

import numpy as np
from symfc.api_symfc import eigh
from symfc.spg_reps.spg_reps_base import SpgRepsBase
from symfc.utils.utils import SymfcAtoms

from pypolymlp.core.data_format import PolymlpStructure


def _set_sgp_reps(st: PolymlpStructure):
    """Set SpgRepsBase instance."""
    symfc_cell = SymfcAtoms(st.types, st.positions.T, st.axis.T)
    spgrep = SpgRepsBase(symfc_cell)
    return spgrep


def compute_spg_projector_O2(st: PolymlpStructure):
    """Compute projector for O2 tensor."""
    spgrep = _set_sgp_reps(st)
    proj = np.zeros((9, 9))
    for r in spgrep._unique_rotations:
        r_c = spgrep._lattice.T @ r @ np.linalg.inv(spgrep._lattice.T)
        proj += np.kron(r_c, r_c)

    proj /= len(spgrep._unique_rotations)
    return proj


def compute_spg_projector_O4(st: PolymlpStructure):
    """Compute projector for O4 tensor."""
    spgrep = _set_sgp_reps(st)
    proj = np.zeros((81, 81))
    for r in spgrep._unique_rotations:
        r_c = spgrep._lattice.T @ r @ np.linalg.inv(spgrep._lattice.T)
        proj += np.kron(np.kron(np.kron(r_c, r_c), r_c), r_c)

    proj /= len(spgrep._unique_rotations)
    return proj


def compute_tensor_basis_O2(st: PolymlpStructure):
    """Compute basis set for O2 tensor."""
    proj = compute_spg_projector_O2(st)
    eigvecs = eigh(proj)
    return eigvecs


def compute_tensor_basis_O4(st: PolymlpStructure):
    """Compute basis set for O4 tensor."""
    proj = compute_spg_projector_O4(st)
    eigvecs = eigh(proj)
    return eigvecs
