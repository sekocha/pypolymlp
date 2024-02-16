#!/usr/bin/env python 
import numpy as np
import math

from symfc.basis_sets.basis_sets_O3 import print_sp_matrix_size
from scipy.sparse import csr_array, kron, coo_array

from symfc.utils.eig_tools import (
    dot_product_sparse,
)

import scipy

def compressed_complement_projector_sum_rules_algo1(
    n_a_compress_mat: csr_array, c_trans: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    r"""Return complementary projection matrix for sum rule compressed by C.

    C = compress_mat.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]).
    C.shape[0] must be equal to NNN333.

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb,kc} = 0

    """
    NNN333 = 27 * N**3
    NN333 = 27 * N**2

    row = np.arange(NNN333)
    col = np.tile(range(NN333), N)
    data = np.zeros(NNN333)
    data[:] = 1 / math.sqrt(N)
    c_sum_cplmt = csr_array((data, (row, col)), shape=(NNN333, NN333))

    if use_mkl:
        n_a_compress_mat = n_a_compress_mat.tocsr()

    print_sp_matrix_size(c_sum_cplmt, " c_sum_cplmt:")
    # bottleneck part
    c_sum_cplmt = dot_product_sparse(c_sum_cplmt.T, c_trans, use_mkl=use_mkl)
    print_sp_matrix_size(c_sum_cplmt, " c_sum_cplmt_compr:")
    print(c_sum_cplmt.data)
    c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
    print_sp_matrix_size(c_sum_cplmt, " c_sum_cplmt_compr:")

    proj_sum_cplmt = dot_product_sparse(
        c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
    )
    # bottleneck part: end
    return proj_sum_cplmt

def compressed_complement_projector_sum_rules_algo4(
    n_a_compress_mat: csr_array, c_trans: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    ''' memory efficient but slow (openmp is not used)'''

    r"""Return complementary projection matrix for sum rule compressed by C.

    C = compress_mat.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]).
    C.shape[0] must be equal to NNN333.

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb,kc} = 0

    """
    NNN333 = 27 * N**3
    NN333 = 27 * N**2

    if use_mkl:
        n_a_compress_mat = n_a_compress_mat.tocsr()

    c_sum_cplmt = csr_array(([], ([], [])), shape=(NN333, c_trans.shape[1]))
    for i in range(N):
        c_sum_cplmt += c_trans[i * NN333 : (i + 1) * NN333]

    print_sp_matrix_size(c_sum_cplmt, " c_sum_cplmt_compr:")

    c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, 
                                    use_mkl=use_mkl)
    print_sp_matrix_size(c_sum_cplmt, " c_sum_cplmt_compr:")

    proj_sum_cplmt = dot_product_sparse(
        c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
    )
    proj_sum_cplmt /= N
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules(
    n_a_compress_mat: csr_array, c_trans: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    """Return complementary projection matrix for sum rule compressed by C."""
    return compressed_complement_projector_sum_rules_algo1(
        n_a_compress_mat, c_trans, N, use_mkl=use_mkl
    )


def compressed_projector_sum_rules(
    n_a_compress_mat: csr_array, c_trans: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = compressed_complement_projector_sum_rules(
        n_a_compress_mat, c_trans, N, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


