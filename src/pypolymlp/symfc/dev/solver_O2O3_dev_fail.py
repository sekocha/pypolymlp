"""Solver of 2nd and 3rd order force constants simultaneously."""
import time

import numpy as np

from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.utils_O2 import get_perm_compr_matrix

from symfc.solvers.solver_funcs import get_batch_slice, solve_linear_equation
from symfc.solvers.solver_O2 import get_training_from_full_basis
from symfc.solvers.solver_O3 import set_2nd_disps

from scipy.sparse import csr_array, coo_array
import gc

from symfc.spg_reps import SpgRepsO1
from symfc.utils.utils_O1 import (
    get_lat_trans_decompr_indices,
    get_lat_trans_compr_matrix,
)


def _NNN333_to_NN33N3(row, N):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3)."""
    # i
    div, rem = np.divmod(row, 27 * (N**2))
    row = div * 27 * (N**2)
    # j
    div, rem = np.divmod(rem, 27 * N)
    row += div * 27 * N 
    # k
    div, rem = np.divmod(rem, 27)
    row += div * 3
    # a
    div, rem = np.divmod(rem, 9)
    row += div * 9 * N
    # b, c
    div, rem = np.divmod(rem, 3)
    row += div * 3 * N + rem
    return row


def reshape_compress_mat(mat, N):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3).

    Return reordered csr_matrix.

    """
    NNN333, nx = mat.shape
    mat = mat.tocoo()
    mat.row = _NNN333_to_NN33N3(mat.row, N)

    '''reshape: (NN33N3,Nx) -> (NN33, N3Nx)'''
    NN33 = (N**2)*9
    N3 = N*3
    mat.row, rem = np.divmod(mat.row, N3)
    mat.col += rem * nx
    return csr_array((mat.data, (mat.row, mat.col)), shape=(NN33, N3*nx))


def _naNN333_to_NN33na3(row, N, n_a):
    """Reorder row indices in a sparse matrix (naNN333->NN33na3)."""
    # i
    div, rem = np.divmod(row, 27 * N * N)
    vec = div * 3
    # j
    div, rem = np.divmod(rem, 27 * N)
    vec += div * 27 * N * n_a
    # k
    div, rem = np.divmod(rem, 27)
    vec += div * 27 * n_a
    # a
    div, rem = np.divmod(rem, 9)
    vec += div
    # b, c
    div, rem = np.divmod(rem, 3)
    vec += div * 9 * n_a + rem * 3 * n_a
    return vec


def reshape_compress_mat_na(mat, N, n_a):
    """Reorder row indices in a sparse matrix (NNN333->NN33N3).

    Return reordered csr_matrix.

    """
    naNN333, nx = mat.shape
    mat1 = mat.tocoo()
    row = _naNN333_to_NN33na3(mat1.row, N, n_a)

    '''reshape: (NN33na3,Nx) -> (NN33, na3Nx)'''
    NN33 = (N**2)*9
    na3 = n_a*3
    mat1 = coo_array((mat1.data, (row, mat1.col)), shape=(naNN333, nx))
    mat1 = mat1.reshape((NN33,-1))
    return mat1.tocsr()

#    row, rem = np.divmod(row, na3)
#    col = mat1.col + rem * nx
#    return csr_array((mat1.data, (row, col)), shape=(NN33, na3*nx))


def get_training_variant(
    supercell,
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    batch_size=200,
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
    N = N3 // 3
    NN33 = N3 * N3
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    n_compr_fc3 = compress_mat_fc3.shape[1]
    n_basis = n_basis_fc2 + n_compr_fc3

    t1 = time.time()
    full_basis_fc2 = compress_mat_fc2 @ compress_eigvecs_fc2
    X2, y_all = get_training_from_full_basis(
        disps, forces, full_basis_fc2.T.reshape((n_basis_fc2, N, N, 3, 3))
    )
    t2 = time.time()
    print(" training data (fc2):    ", t2 - t1)

    t1 = time.time()
    c_perm_fc2 = get_perm_compr_matrix(N)
    '''
    compress_mat_fc3 = (
        csr_NNN333_to_NN33N3(compress_mat_fc3, N).reshape((NN33, -1)).tocsr()
    )
    '''
    n_a = compress_mat_fc3.shape[0] // (27*(N**2))
    na3 = n_a * 3
    compress_mat_fc3_reshape = reshape_compress_mat_na(compress_mat_fc3, N, n_a)
    compress_mat_fc3_reshape = -0.5 * (c_perm_fc2.T @ compress_mat_fc3_reshape)
    t2 = time.time()
    print(" precond. compress_mat (for fc3):", t2 - t1)

    spg_reps = SpgRepsO1(supercell)
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    c_trans_O1 = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)

    sparse_disps = True if use_mkl else False
    mat33 = np.zeros((n_compr_fc3, n_compr_fc3), dtype=float)
    mat23 = np.zeros((n_basis_fc2, n_compr_fc3), dtype=float)
    mat3y = np.zeros(n_compr_fc3, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        batch_size = end - begin
        t01 = time.time()
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        disps_batch = disps_batch @ c_perm_fc2
        X3_compr = dot_product_sparse(
            disps_batch, compress_mat_fc3_reshape, use_mkl=use_mkl, dense=True
        ).reshape((-1, n_compr_fc3))

        X3 = np.zeros((batch_size * N3, n_compr_fc3))
        for i_st in range(batch_size):
            X3[i_st * N3 : (i_st + 1) * N3] = (
                c_trans_O1 @ X3_compr[i_st * na3 : (i_st + 1) * na3]
            )
        y_batch = forces[begin:end].reshape(-1)
        mat23 += X2[begin * N3 : end * N3].T @ X3
        mat33 += X3.T @ X3
        mat3y += X3.T @ y_batch
        t02 = time.time()
        print(" solver_block:", end, ":, t =", t02 - t01)

    XTX = np.zeros((n_basis, n_basis), dtype=float)
    XTy = np.zeros(n_basis, dtype=float)
    XTX[:n_basis_fc2, :n_basis_fc2] = X2.T @ X2
    XTX[:n_basis_fc2, n_basis_fc2:] = mat23
    XTX[n_basis_fc2:, :n_basis_fc2] = XTX[:n_basis_fc2, n_basis_fc2:].T
    XTX[n_basis_fc2:, n_basis_fc2:] = mat33
    XTy[:n_basis_fc2] = X2.T @ y_all
    XTy[n_basis_fc2:] = mat3y

    t3 = time.time()
    print(" (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):", t3 - t2)
    return XTX, XTy


def run_solver_O2O3_variant(
    supercell,
    disps,
    forces,
    compress_mat_fc2,
    compress_mat_fc3,
    compress_eigvecs_fc2,
    batch_size=200,
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
    XTX, XTy = get_training_variant(
        supercell,
        disps,
        forces,
        compress_mat_fc2,
        compress_mat_fc3,
        compress_eigvecs_fc2,
        batch_size=batch_size,
        use_mkl=use_mkl,
    )
    coefs = solve_linear_equation(XTX, XTy)
    n_basis_fc2 = compress_eigvecs_fc2.shape[1]
    coefs_fc2, coefs_fc3 = coefs[:n_basis_fc2], coefs[n_basis_fc2:]
    return coefs_fc2, coefs_fc3
