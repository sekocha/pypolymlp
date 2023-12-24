#!/usr/bin/env python
import numpy as np
import time

from scipy.sparse import csr_matrix

from pypolymlp.symfc_dev.linalg import dot_sp_mats
from pypolymlp.symfc_dev.solver import fit, solve_linear_equation

def set_2nd_disps(disps, sparse=True):
    '''
    Calculate Kronecker products of displacements
    disps: (n_samples, N3)
    disps_2nd: (n_samples, NN33)
    '''
    N = disps.shape[1] // 3
    n_samples = disps.shape[0]
    disps_2nd = np.zeros((n_samples, 9*(N**2)))
    for i, u_vec in enumerate(disps):
        u2 = np.kron(u_vec, u_vec).reshape((N,3,N,3))
        disps_2nd[i] = u2.transpose((0,2,1,3)).reshape(-1)

    '''algorithm 2
    n_samples = disps.shape[0]
    disps_2nd = disps[:,:,None] @ disps[:,None,:]
    disps_2nd = disps_2nd.reshape(n_samples,N,3,N,3)\
            .transpose((0,1,3,2,4)).reshape((n_samples,-1))
    '''
    if sparse:
        return csr_matrix(disps_2nd)
    return disps_2nd


def get_training_from_full_basis_set(disps, forces, full_basis):
    '''
    Training dataset (X, y) transformed from displacements, 
    forces, and full basis-set.

    disps: (n_samples, N3)
    forces: (n_samples, N3)
    full_basis: (n_basis, N, N, N, 3, 3, 3)

    X: features (calculated from Kronecker products of displacements)
                (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)

    X = displacements @ compress_mat @ compress_eigvecs

    displacements: (n_samples, NN33)
    compress_mat: (NNN333, n_compr)
    compress_eigvecs: (n_compr, n_basis)
    Matrix reshapings are appropriately applied to compress_mat 
    and its products.
    '''

    N3 = full_basis.shape[1] * 3
    NN33 = N3 * N3
    n_basis = full_basis.shape[0]

    disps_reshape = set_2nd_disps(disps, sparse=False)
       
    full_basis = full_basis.transpose((1,2,4,5,3,6,0)).reshape((NN33,-1))
    X = - 0.5 * np.dot(disps_reshape, full_basis)
    X = X.reshape((-1,n_basis))
    y = forces.reshape(-1)
    return X, y

def run_solver_dense_O3(disps, forces, compress_mat, compress_eigvecs):
    '''
    Estimating coeffs. in X @ coeffs = y 
    X: features (calculated from Kronecker products of displacements)
                (n_samples * N3, N_basis)
    y: observations (forces), (n_samples * N3)
    '''

    N3 = disps.shape[1] 
    N = N3 // 3

    t0 = time.time()
    full_basis = compress_mat @ compress_eigvecs
    n_basis = full_basis.shape[1]
    full_basis = full_basis.T.reshape((n_basis,N,N,N,3,3,3))
    t1 = time.time()
    print(' elapsed time (compute full basis) =', t1-t0)

    X, y = get_training_from_full_basis_set(disps, forces, full_basis)
    coefs = fit(X, y)
    t2 = time.time()
    print(' elapsed time (solve fc3)          =', t2-t1)

    return coefs

def NNN333_to_NN33N3(row, N):
    ''' Reorder row indices in a sparse matrix (NNN333->NN33N3).
    '''
    i, rem = np.divmod(row, 27*(N**2))
    j, rem = np.divmod(rem, 27*N)
    k, rem = np.divmod(rem, 27)
    a, rem = np.divmod(rem, 9)
    b, c = np.divmod(rem, 3)

    vec = i * 27*(N**2)
    vec += j * 27*N
    vec += k * 3
    vec += a * 9*N
    vec += b * 3*N + c
    '''
    # slow
    ijkabc = np.zeros((row.shape[0],6))
    ijkabc[:,0], rem = np.divmod(row, 27*(N**2))
    ijkabc[:,1], rem = np.divmod(rem, 27*N)
    ijkabc[:,2], rem = np.divmod(rem, 27)
    ijkabc[:,3], rem = np.divmod(rem, 9)
    ijkabc[:,4], ijkabc[:,5] = np.divmod(rem, 3)
    coeffs = np.array([27*(N**2),27*N,3,9*N,3*N,1])
    vec = ijkabc @ coeffs
    '''
    return vec

def csr_NNN333_to_NN33N3(mat, N):
    ''' Reorder row indices in a sparse matrix (NNN333->NN33N3).
        Return reordered csr_matrix.
    '''
    NNN333, nx = mat.shape
    mat = mat.tocoo()
    row = NNN333_to_NN33N3(mat.row, N)
    mat = csr_matrix((mat.data, (row, mat.col)), shape=(NNN333,nx))
    return mat

def get_batch_slice(n_data, batch_size):
    ''' Calculate slice indices for a given batch size.
    '''

    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch

def get_training_exact(disps, 
                       forces, 
                       compress_mat, 
                       compress_eigvecs,
                       use_mkl=True,
                       batch_size=200):
    '''
    Calculating X.T @ X and X.T @ y.
    X = displacements @ compress_mat @ compress_eigvecs

    displacements: (n_samples, NN33)
    compress_mat: (NNN333, n_compr)
    compress_eigvecs: (n_compr, n_basis)
    Matrix reshapings are appropriately applied to compress_mat 
    and its products.

    X.T @ X and X.T @ y are sequentially calculated using divided dataset.
        X.T @ X = \sum_i X_i.T @ X_i 
        X.T @ y = \sum_i X_i.T @ y_i (i: batch index)
    '''

    N3 = disps.shape[1] 
    N = N3 // 3
    NN33 = N3 * N3
    n_compr = compress_mat.shape[1]

    t1 = time.time()
    compress_mat = -0.5 * csr_NNN333_to_NN33N3(compress_mat, N)\
                                        .reshape((NN33,-1)).tocsr()
    t2 = time.time()
    print(' reshape(compr):   ', t2-t1)

    sparse_disps = True if use_mkl else False
    XTX = np.zeros((n_compr, n_compr), dtype=float)
    XTy = np.zeros(n_compr, dtype=float)
    begin_batch, end_batch = get_batch_slice(disps.shape[0], batch_size)
    for begin, end in zip(begin_batch, end_batch):
        t01 = time.time()
        disps_batch = set_2nd_disps(disps[begin:end], sparse=sparse_disps)
        X = dot_sp_mats(disps_batch, 
                        compress_mat, 
                        mkl=use_mkl, 
                        dense=True).reshape((-1,n_compr))
        y_batch = forces[begin:end].reshape(-1)
        XTX += X.T @ X
        XTy += X.T @ y_batch
        t02 = time.time()
        print(' solver_block:', end, ':, t =', t02-t01)

    XTX = compress_eigvecs.T @ XTX @ compress_eigvecs
    XTy = compress_eigvecs.T @ XTy
    t3 = time.time()
    print(' (disp @ compr @ eigvecs).T @ (disp @ compr @ eigvecs):', t3-t2)
    return XTX, XTy

def run_solver_sparse_O3(disps, 
                         forces, 
                         compress_mat, 
                         compress_eigvecs,
                         use_mkl=True,
                         batch_size=200):
    '''
    Estimating coeffs. in X @ coeffs = y 
      by solving normal equation (X.T @ X) @ coeffs = X.T @ y
        X = displacements @ compress_mat @ compress_eigvecs
        Matrix reshapings are appropriately applied.
        X: features (n_samples * N3, N_basis)
        y: observations (forces), (n_samples * N3)
    '''
    XTX, XTy = get_training_exact(disps, 
                                  forces, 
                                  compress_mat, 
                                  compress_eigvecs,
                                  use_mkl=use_mkl,
                                  batch_size=batch_size)
    coefs = solve_linear_equation(XTX, XTy)
    return coefs


