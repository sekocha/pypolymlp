"""Utilities to calculate fourth-order tensor properties."""

import itertools

import numpy as np
from symfc.api_symfc import eigh

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.symfc_utils import set_spg_reps


def compute_projector_O4(st: PolymlpStructure):
    """Compute projector for O4 tensor."""
    proj_perm = compute_perm_projector_O4(st)
    proj_spg = compute_spg_projector_O4(st)
    return proj_perm @ proj_spg


def compute_spg_projector_O4(st: PolymlpStructure):
    """Compute projector for O4 tensor."""
    spgrep = set_spg_reps(st)
    proj = np.zeros((81, 81))
    for r in spgrep._unique_rotations:
        r_c = spgrep._lattice.T @ r @ np.linalg.inv(spgrep._lattice.T)
        proj += np.kron(np.kron(np.kron(r_c, r_c), r_c), r_c)

    proj /= len(spgrep._unique_rotations)
    return proj


def compute_perm_projector_O4(st: PolymlpStructure):
    """Compute projector for permutations in O4 tensor."""
    comb = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    size = 81
    perm_basis = []
    for i1, i2 in itertools.combinations_with_replacement(comb, 2):
        cand = [
            (i1[0], i1[1], i2[0], i2[1]),
            (i1[0], i1[1], i2[1], i2[0]),
            (i1[1], i1[0], i2[0], i2[1]),
            (i1[1], i1[0], i2[1], i2[0]),
            (i2[0], i2[1], i1[0], i1[1]),
            (i2[0], i2[1], i1[1], i1[0]),
            (i2[1], i2[0], i1[0], i1[1]),
            (i2[1], i2[0], i1[1], i1[0]),
        ]
        uniq = np.array(list(set(cand)))
        val = 1 / np.sqrt(uniq.shape[0])
        basis = np.zeros(size)
        indices = uniq[:, 0] * 27 + uniq[:, 1] * 9 + uniq[:, 2] * 3 + uniq[:, 3]
        basis[indices] = val
        perm_basis.append(basis)

    perm_basis = np.array(perm_basis).T
    return perm_basis @ perm_basis.T


def compute_tensor_basis_O4(st: PolymlpStructure):
    """Compute basis set for O4 tensor."""
    proj = compute_projector_O4(st)
    eigvecs = eigh(proj)
    return eigvecs
