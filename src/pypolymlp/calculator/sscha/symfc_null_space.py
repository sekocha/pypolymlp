"""Utilities for applying null space to Symfc."""

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from symfc import Symfc
from symfc.api_symfc import eigh_projector


def _reshape_proj_kron(basis: np.ndarray, N: int):
    """Reshape Kronecker product of N3 projector."""
    n_basis = basis.shape[1]
    return (
        basis.reshape((N, 3, N, 3, -1))
        .transpose((0, 2, 1, 3, 4))
        .reshape((-1, n_basis))
    )


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

    null_kron1 = np.kron(null_space_basis, null_space_basis)
    null_kron1 = _reshape_proj_kron(null_kron1, N)

    null_kron2 = np.kron(np.eye(N3), null_space_basis)
    null_kron2 = _reshape_proj_kron(null_kron2, N)

    prod = null_kron1.T @ compr @ eigvecs_core
    proj_compr1 = prod.T @ prod
    prod = null_kron2.T @ compr @ eigvecs_core
    proj_compr2 = prod.T @ prod
    proj_compr = np.eye(proj_compr1.shape[0]) + proj_compr1 - 2 * proj_compr2
    eigvecs = eigh_projector(proj_compr)

    # TODO: Revise Symfc, implement setter.
    symfc.basis_set[2]._blocked_basis_set = eigvecs_core @ eigvecs
    return symfc


def apply_null_space_stable(symfc: Symfc, null_space_basis: np.ndarray):
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

    proj_Z = np.eye(null_space_basis.shape[0]) - null_space_basis @ null_space_basis.T
    proj_Z = csr_matrix(proj_Z)

    proj_kron = scipy.sparse.kron(proj_Z, proj_Z).toarray()
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
