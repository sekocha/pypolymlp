"""Utilities for applying null space to Symfc."""

import numpy as np
import scipy

# from scipy.sparse import csr_matrix
from symfc import Symfc
from symfc.api_symfc import eigh_projector


def apply_null_space(symfc: Symfc, null_space_basis: np.ndarray):
    """Apply null space to FC2 basis set.

    Parameters
    ----------
    null_space_basis: shape=(N3, n_null).
    """
    # TODO: Efficient procedure.
    basis_set = symfc.basis_set[2]
    compr = basis_set.compression_matrix
    eigvecs_core = basis_set.basis_set

    Z = scipy.linalg.null_space(null_space_basis.T)
    proj_Z = Z @ Z.T

    N3, n_null = null_space_basis.shape
    N = N3 // 3
    NN33 = N**2 * 9

    proj_kron = np.kron(proj_Z, proj_Z)
    proj_kron = (
        proj_kron.reshape((N, 3, N, 3, N, 3, N, 3))
        .transpose((0, 6, 1, 7, 2, 4, 3, 5))
        .reshape((NN33, NN33))
    )
    proj_compr = eigvecs_core.T @ (compr.T @ proj_kron @ compr) @ eigvecs_core
    eigvecs = eigh_projector(proj_compr)

    # TODO: Revise Symfc, implement setter.
    symfc.basis_set[2]._blocked_basis_set = eigvecs_core @ eigvecs
    return symfc
