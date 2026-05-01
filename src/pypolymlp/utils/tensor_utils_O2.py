"""Utilities to calculate second-order tensor properties."""

import numpy as np
from symfc.api_symfc import eigh

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.symfc_utils import set_spg_reps


def compute_projector_O2(st: PolymlpStructure):
    """Compute projector for O2 tensor."""
    proj_perm = compute_perm_projector_O2(st)
    proj_spg = compute_spg_projector_O2(st)
    return proj_perm @ proj_spg


def compute_spg_projector_O2(st: PolymlpStructure):
    """Compute projector for O2 tensor."""
    spgrep = set_spg_reps(st)
    proj = np.zeros((9, 9))
    for r in spgrep._unique_rotations:
        r_c = spgrep._lattice.T @ r @ np.linalg.inv(spgrep._lattice.T)
        proj += np.kron(r_c, r_c)

    proj /= len(spgrep._unique_rotations)
    return proj


def compute_perm_projector_O2(st: PolymlpStructure):
    """Compute projector for permutations in O2 tensor."""
    comb = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    size = 9
    perm_basis = []
    for i, j in comb:
        uniq = np.array(list({(i, j), (j, i)}))
        val = 1 / np.sqrt(len(uniq))
        basis = np.zeros(size)
        indices = uniq[:, 0] * 3 + uniq[:, 1]
        basis[indices] = val
        perm_basis.append(basis)

    perm_basis = np.array(perm_basis).T
    return perm_basis @ perm_basis.T


def compute_tensor_basis_O2(st: PolymlpStructure):
    """Compute basis set for O2 tensor."""
    proj = compute_spg_projector_O2(st)
    eigvecs = eigh(proj)
    return eigvecs
