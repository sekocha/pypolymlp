"""Utilities for applying null space to Symfc."""

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from symfc import Symfc
from symfc.api_symfc import eigh_projector


def apply_null_space(symfc: Symfc, null_space_basis: np.ndarray):
    """Apply null space to FC2 basis set.

    Parameters
    ----------
    null_space_basis: shape=(N3, n_null).
    """
    basis_set = symfc.basis_set[2]
    compr = basis_set.compression_matrix
    eigvecs_core = basis_set.basis_set

    N3, n_null = null_space_basis.shape
    N = N3 // 3
    NN33 = N**2 * 9

    ids_jb = ((np.arange(N) * 9)[:, None] + np.arange(3)[None, :]).reshape(-1)

    trial_algo = 2
    if trial_algo == 1:
        data = np.tile(null_space_basis.reshape(-1), N3)
        cols = np.repeat(np.arange(N3 * n_null), N3)
        rows = []
        for i in range(N):
            for a in range(3):
                add = i * (N * 9) + a * 3
                rows.extend(ids_jb + add)
                rows.extend(ids_jb + add)
        rows = np.array(rows)
        C = csr_matrix((data, (rows, cols)), shape=(NN33, N3 * n_null))
    elif trial_algo == 2:
        data = np.tile(null_space_basis.reshape(-1), N)
        cols = np.repeat(np.arange(N * n_null), N3)
        rows = []
        for i in range(N):
            add = i * (N * 9)
            rows.extend(ids_jb + add)
            add = i * (N * 9) + 3
            rows.extend(ids_jb + add)
        rows = np.array(rows)
        C = csr_matrix((data, (rows, cols)), shape=(NN33, N * n_null))

    # offsets = (
    #     np.arange(N)[:, None] * (N**2 * 3)
    #     + np.arange(3)[None, :] * 3
    # )
    # rows = (ids_jb[None, None, :] + offsets[:, :, None]).reshape(-1)

    proj = scipy.sparse.eye(NN33) - C @ C.T
    proj_compr = eigvecs_core.T @ (compr.T @ proj @ compr) @ eigvecs_core
    eigvecs = eigh_projector(proj_compr)
    print(eigvecs.shape)

    # TODO: Revise Symfc, implement setter.
    symfc.basis_set[2]._blocked_basis_set = eigvecs_core @ eigvecs
    return symfc
