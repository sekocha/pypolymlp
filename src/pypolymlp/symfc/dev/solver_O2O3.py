"""Solver of 2nd and 3rd order force constants simultaneously."""

import time

import numpy as np
from scipy.sparse import csr_array
from symfc.solvers.solver_O2 import get_training_from_full_basis
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.utils.utils_O3 import get_lat_trans_decompr_indices_O3


def set_2nd_disps(disps, sparse=True):
    """Calculate Kronecker products of displacements.

    Parameter
    ---------
    disps: shape=(n_supercell, N3)

    Return
    ------
    disps_2nd: shape=(n_supercell, NN33)
    """
    n_supercell = disps.shape[0]
    N = disps.shape[1] // 3
    disps_2nd = (disps[:, :, None] * disps[:, None, :]).reshape((-1, N, 3, N, 3))
    disps_2nd = disps_2nd.transpose((0, 1, 3, 2, 4)).reshape((n_supercell, -1))
    """
    N = disps.shape[1] // 3
    disps_2nd = np.zeros((disps.shape[0], 9 * (N**2)))
    for i, u_vec in enumerate(disps):
        u2 = np.kron(u_vec, u_vec).reshape((N, 3, N, 3))
        disps_2nd[i] = u2.transpose((0, 2, 1, 3)).reshape(-1)
    """
    if sparse:
        return csr_array(disps_2nd)
    return disps_2nd


# def _nNN333_to_NN33n3(row, N, n):
#     """Reorder row indices in a sparse matrix (nNN333->NN33n3)."""
#     row, rem = np.divmod(row, 27 * N * N)
#     row *= 3
#     div, rem = np.divmod(rem, 27 * N)
#     row += div * 27 * N * n
#     div, rem = np.divmod(rem, 27)
#     row += div * 27 * n
#     div, rem = np.divmod(rem, 9)
#     row += div
#     div, rem = np.divmod(rem, 3)
#     row += div * 9 * n
#     row += rem * 3 * n
#     return row
#
#
# def csr_nNN333_to_NN33n3(mat, N, n):
#     """Reorder row indices in a sparse matrix (NNn333->NN33n3).
#
#     Return reordered csr_matrix.
#
#     """
#     nNN333, nx = mat.shape
#     row, col = mat.nonzero()
#     row = _nNN333_to_NN33n3(row, N, n)
#     mat = csr_array((mat.data, (row, col)), shape=(nNN333, nx))
#     return mat


def csr_nNN333_to_NN33_n3nx(mat, N, n):
    """Reorder and reshape a sparse matrix (nNN333,nx)->(NN33,n3nx).

    Return reordered csr_matrix.
    """
    _, nx = mat.shape
    NN33 = N**2 * 9
    n3nx = n * 3 * nx
    row, col = mat.nonzero()

    div, rem = np.divmod(row, 27 * N * N)
    col += div * 3 * nx
    div, rem = np.divmod(rem, 27 * N)
    row = div * 9 * N
    div, rem = np.divmod(rem, 27)
    row += div * 9
    div, rem = np.divmod(rem, 9)
    col += div * nx
    div, rem = np.divmod(rem, 3)
    row += div * 3 + rem

    return csr_array((mat.data, (row, col)), shape=(NN33, n3nx))


def prepare_normal_equation(
    disps,
    forces,
    compress_mat_fc2,
    compact_compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    trans_perms,
    batch_size=100,
    use_mkl=False,
):
    r"""Calculate X.T @ X and X.T @ y.

    X = displacements @ compress_mat @ compress_eigvecs
    X = np.hstack([X_fc2, X_fc3])

    displacements (fc2): (n_samples, N3)
    displacements (fc3): (n_samples, NN33)
    compress_mat_fc2: (NN33, n_compr)
    compress_mat_fc3: (NNN333, n_compr_fc3)
    compress_eigvecs_fc2: (n_compr_fc2, n_basis_fc2)
    compress_eigvecs_fc3: (n_compr_fc3, n_basis_fc3)
    Matrix reshapings are appropriately applied to compress_mat
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
    X.T @ X = \sum_i X_i.T @ X_i
    X.T @ y = \sum_i X_i.T @ y_i (i: batch index)
    """
    N3 = disps.shape[1]
    NN333 = N3 * N3 * 3
    N = N3 // 3
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    n_basis_fc3 = compress_eigvecs_fc3.shape[1]
    n_compr_fc3 = compact_compress_mat_fc3.shape[1]
    n_basis = n_basis_fc2 + n_basis_fc3

    t1 = time.time()
    full_basis_fc2 = compress_mat_fc2 @ compress_eigvecs_fc2
    X2, y_all = get_training_from_full_basis(
        disps, forces, full_basis_fc2.T.reshape((n_basis_fc2, N, N, 3, 3))
    )
    t2 = time.time()
    print("Solver_normal_equation (FC2):    ", t2 - t1)

    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    n_lp, _ = trans_perms.shape

    size_compr = len(decompr_idx) * compact_compress_mat_fc3.shape[1]
    if size_compr > 1e14:
        batch_size_atom = N // 4
    elif size_compr > 1e13:
        batch_size_atom = N // 2
    else:
        batch_size_atom = N
    begin_batch_atom, end_batch_atom = get_batch_slice(N, batch_size_atom)

    t_all1 = time.time()
    for begin_i, end_i in zip(begin_batch_atom, end_batch_atom):
        print("Solver_atoms:", begin_i, "-", end_i - 1)
        n_atom_batch = end_i - begin_i
        """
        c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
        compr_mat = (
            c_trans[i_atom * NN333 : (i_atom + 1) * NN333] @ compact_compress_mat_fc3
        )
        compr_mat = (
            - 0.5 * csr_nNN333_to_NN33n3(compr_mat, N, 1).reshape((NN33, -1)).tocsr()
        )
        """
        t1 = time.time()
        const = -0.5 / np.sqrt(n_lp)
        compr_mat = compact_compress_mat_fc3[
            decompr_idx[begin_i * NN333 : end_i * NN333]
        ]
        compr_mat = const * csr_nNN333_to_NN33_n3nx(compr_mat, N, n_atom_batch)
        t2 = time.time()
        print("Solver_compr_matrix_reshape:, t =", t2 - t1)

        for begin, end in zip(begin_batch, end_batch):
            t01 = time.time()
            disps_batch = set_2nd_disps(disps[begin:end], sparse=False)
            X3 = dot_product_sparse(
                disps_batch, compr_mat, use_mkl=use_mkl, dense=True
            ).reshape((-1, n_compr_fc3))

            X2_ids = np.array(
                [
                    i * N3 + j * 3 + a
                    for i in range(begin, end)
                    for j in range(begin_i, end_i)
                    for a in range(3)
                ]
            )
            mat23 += X2[X2_ids].T @ X3
            mat33 += X3.T @ X3

            y_batch = forces[begin:end]
            mat3y += X3.T @ y_batch[:, begin_i * 3 : end_i * 3].reshape(-1)

            t02 = time.time()
            print("Solver_block:", end, ":, t =", t02 - t01)

    XTX = np.zeros((n_basis, n_basis), dtype=float)
    XTy = np.zeros(n_basis, dtype=float)
    XTX[:n_basis_fc2, :n_basis_fc2] = X2.T @ X2
    XTX[:n_basis_fc2, n_basis_fc2:] = mat23 @ compress_eigvecs_fc3
    XTX[n_basis_fc2:, :n_basis_fc2] = XTX[:n_basis_fc2, n_basis_fc2:].T
    XTX[n_basis_fc2:, n_basis_fc2:] = (
        compress_eigvecs_fc3.T @ mat33 @ compress_eigvecs_fc3
    )
    XTy[:n_basis_fc2] = X2.T @ y_all
    XTy[n_basis_fc2:] = compress_eigvecs_fc3.T @ mat3y

    t_all2 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t_all2 - t_all1)
    return XTX, XTy


def run_solver_O2O3(
    disps,
    forces,
    compress_mat_fc2,
    compact_compress_mat_fc3,
    compress_eigvecs_fc2,
    compress_eigvecs_fc3,
    trans_perms,
    batch_size=100,
    use_mkl=False,
):
    """Estimate coeffs. in X @ coeffs = y.

    X_fc2 = displacements_fc2 @ compress_mat_fc2 @ compress_eigvecs_fc2
    X_fc3 = displacements_fc3 @ compress_mat_fc3 @ compress_eigvecs_fc3
    X = np.hstack([X_fc2, X_fc3])

    Matrix reshapings are appropriately applied.
    X: features (n_samples * N3, N_basis_fc2 + N_basis_fc3)
    y: observations (forces), (n_samples * N3)

    """
    XTX, XTy = prepare_normal_equation(
        disps,
        forces,
        compress_mat_fc2,
        compact_compress_mat_fc3,
        compress_eigvecs_fc2,
        compress_eigvecs_fc3,
        trans_perms,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3
