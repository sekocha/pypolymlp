"""Utilities for applying null space to Symfc."""

import numpy as np
import scipy
from scipy.sparse import csr_array
from symfc import Symfc
from symfc.api_symfc import eigh_projector
from symfc.eig_solvers.matrix import BlockMatrixNode


def _reshape_null_kron(basis: np.ndarray, N: int):
    """Reshape Kronecker product of N3 projector."""
    n_basis = basis.shape[1]
    return (
        basis.reshape((N, 3, N, 3, -1))
        .transpose((0, 2, 1, 3, 4))
        .reshape((-1, n_basis))
    )


def _reshape_null_kron_csr(basis: csr_array, N: int):
    """Reshape sparse Kronecker product of N3 projector."""
    row, col = basis.nonzero()
    new_row, rem = np.divmod(row, 9 * N)
    new_row *= 9 * N
    alpha, rem = np.divmod(rem, 3 * N)
    new_row += alpha * 3
    j, beta = np.divmod(rem, 3)
    new_row += j * 9 + beta
    return csr_array((basis.data, (new_row, col)), shape=basis.shape)


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
    null_kron1 = _reshape_null_kron(null_kron1, N)

    null_kron2 = scipy.sparse.kron(scipy.sparse.eye(N3), csr_array(null_space_basis))
    null_kron2 = _reshape_null_kron_csr(null_kron2, N)

    prod = (null_kron1.T @ compr) @ eigvecs_core
    proj_compr1 = prod.T @ prod

    prod = (null_kron2.T @ compr) @ eigvecs_core
    proj_compr2 = prod.T @ prod

    proj_compr = np.eye(proj_compr1.shape[0]) + proj_compr1 - 2 * proj_compr2
    eigvecs = eigh_projector(proj_compr)

    eigvecs = eigvecs_core @ eigvecs
    n_rows, n_cols = eigvecs.shape
    block = BlockMatrixNode(
        rows=np.arange(n_rows),
        col_begin=0,
        col_end=n_cols,
        data=eigvecs,
        index=0,
    )
    symfc.basis_set[2]._blocked_basis_set = block
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
    proj_Z = csr_array(proj_Z)

    proj_kron = scipy.sparse.kron(proj_Z, proj_Z).toarray()
    proj_kron = (
        proj_kron.reshape((N, 3, N, 3, N, 3, N, 3))
        .transpose((0, 6, 1, 7, 2, 4, 3, 5))
        .reshape((NN33, NN33))
    )
    proj_compr = eigvecs_core.T @ (compr.T @ proj_kron @ compr) @ eigvecs_core
    eigvecs = eigh_projector(proj_compr)

    eigvecs = eigvecs_core @ eigvecs
    n_rows, n_cols = eigvecs.shape
    block = BlockMatrixNode(
        rows=np.arange(n_rows),
        col_begin=0,
        col_end=n_cols,
        data=eigvecs,
        index=0,
    )
    symfc.basis_set[2]._blocked_basis_set = block
    return symfc


def get_supercell_null_space(null_space_basis: np.ndarray, n_unitcells: int):
    """Convert null space basis in unit cell to basis in supercell."""
    sup_null_space_basis = []
    for basis_element in null_space_basis.T:
        tmp_basis = []
        tmp = basis_element.reshape((-1, 3))
        for t in tmp:
            for i in range(n_unitcells):
                tmp_basis.extend(t)
        sup_null_space_basis.append(tmp_basis)
    sup_null_space_basis = np.array(sup_null_space_basis).T
    null_space_basis_supercell, _ = np.linalg.qr(sup_null_space_basis)
    return null_space_basis_supercell


def recover_fc2(symfc: Symfc, coefs: np.ndarray):
    """Recover FC2 from coefficients for basis set."""
    basis_set = symfc.basis_set[2]
    compr_mat = basis_set.compression_matrix

    n_atom = np.sqrt(compr_mat.shape[0] // 9)
    fc = basis_set.basis_set @ coefs
    return (compr_mat @ fc).reshape((n_atom, n_atom, 3, 3))


def project_fc2_space(symfc: Symfc, fc2_: np.ndarray):
    """Apply basis set to force constants.

    The projected force constants are calculated as follows.
    fc2_projected = P @ fc2
                  = (C @ B) @ (C @ B).T @ fc2
                  = C @ B @ B.T @ C.T @ fc2
    """
    if symfc is None:
        raise RuntimeError("Symfc instance not found.")

    N = fc2_.shape[0]
    B = symfc.basis_set[2].basis_set
    C = symfc.basis_set[2].compression_matrix

    fc2 = C.T @ fc2_.reshape(-1)
    fc2 = B.T @ fc2
    fc2 = B @ fc2
    fc2 = C @ fc2
    fc2 = fc2.reshape((N, N, 3, 3))
    return fc2
