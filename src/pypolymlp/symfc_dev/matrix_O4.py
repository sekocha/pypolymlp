#!/usr/bin/env python
import numpy as np
import itertools
from math import sqrt
import time

import scipy
from scipy.sparse import csr_matrix, coo_matrix

from pypolymlp.symfc_dev.linalg import dot_sp_mats

def N3N3N3N3_to_NNNN3333(combinations_perm, N):
    vec = combinations_perm[:,0]  // 3 * 81*N**3 
    vec += combinations_perm[:,1] // 3 * 81*N**2
    vec += combinations_perm[:,2] // 3 * 81*N
    vec += combinations_perm[:,3] // 3 * 81
    vec += combinations_perm[:,0] % 3 * 27 
    vec += combinations_perm[:,1] % 3 * 9 
    vec += combinations_perm[:,2] % 3 * 3 
    vec += combinations_perm[:,3] % 3 
    return vec
 
def permutation_symmetry_basis(N):
    '''Return compression matrix by permutation symmetry.

    Matrix shape is (NNNN3333, ???).
    todo: permutation_symmetry_basis is expected to be 
          accelerated using C implementation
    '''
    NNNN3333 = 81*N**4
    combinations4 = np.array(list(itertools.combinations(range(3*N),4)))
    combinations3 = np.array(list(itertools.combinations(range(3*N),3)))
    combinations2 = np.array(list(itertools.combinations(range(3*N),2)))
    combinations1 = np.array([[i,i,i,i] for i in range(3*N)])

    n_col4 = combinations4.shape[0]
    n_col3 = combinations3.shape[0] * 3
    n_col2_1 = combinations2.shape[0] * 2
    n_col2_2 = combinations2.shape[0]
    n_col1 = combinations1.shape[0]
    n_col = n_col4 + n_col3 + n_col2_1 + n_col2_2 + n_col1
    n_data4 = combinations4.shape[0] * 24
    n_data3 = combinations3.shape[0] * 36
    n_data2_1 = combinations2.shape[0] * 8
    n_data2_2 = combinations2.shape[0] * 6
    n_data1 = combinations1.shape[0]
    n_data = n_data4 + n_data3 + n_data2_1 + n_data2_2 + n_data1

    row = np.zeros(n_data, dtype="int_")
    col = np.zeros(n_data, dtype="int_")
    data = np.zeros(n_data, dtype=float)

    # (4)
    begin_id, end_id = 0, n_data4
    perms = np.array(list(itertools.permutations([0,1,2,3])))
    combinations_perm = combinations4[:,perms].reshape((-1,4))
    row[begin_id:end_id] = N3N3N3N3_to_NNNN3333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col4), 24)
    data[begin_id:end_id] = 1/sqrt(24)

    # (3)
    begin_id = end_id
    end_id = begin_id + n_data3
    perms = np.vstack([
        np.array(list(set(itertools.permutations([0,0,1,2])))),
        np.array(list(set(itertools.permutations([0,1,1,2])))),
        np.array(list(set(itertools.permutations([0,1,2,2]))))
        ])
    combinations_perm = combinations3[:,perms].reshape((-1,4))
    row[begin_id:end_id] = N3N3N3N3_to_NNNN3333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col4, n_col4 + n_col3), 12)
    data[begin_id:end_id] = 1/sqrt(12)

    # (2-1)
    begin_id = end_id
    end_id = begin_id + n_data2_1
    perms = np.vstack([
        np.array(list(set(itertools.permutations([0,0,0,1])))),
        np.array(list(set(itertools.permutations([0,1,1,1]))))
        ])

    combinations_perm = combinations2[:,perms].reshape((-1,4))
    row[begin_id:end_id] = N3N3N3N3_to_NNNN3333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col4 + n_col3, 
                                           n_col4 + n_col3 + n_col2_1), 4)
    data[begin_id:end_id] = 1/sqrt(4)

    # (2-2)
    begin_id = end_id
    end_id = begin_id + n_data2_2
    perms = np.array(list(set(itertools.permutations([0,0,1,1]))))
    combinations_perm = combinations2[:,perms].reshape((-1,4))
    row[begin_id:end_id] = N3N3N3N3_to_NNNN3333(combinations_perm, N)
    col[begin_id:end_id] = np.repeat(range(n_col4 + n_col3 + n_col2_1, 
                                    n_col4 + n_col3 + n_col2_1 + n_col2_2), 6)
    data[begin_id:end_id] = 1/sqrt(6)

    # (1)
    begin_id = end_id
    row[begin_id:] = N3N3N3N3_to_NNNN3333(combinations1, N)
    col[begin_id:] = np.array(range(
                        n_col4 + n_col3 + n_col2_1 + n_col2_2, n_col))
    data[begin_id:] = 1.0

    return coo_matrix((data, (row, col)), shape=(NNNN3333, n_col))

def compressed_complement_projector_sum_rules_algo1(C, N, mkl=False):
    '''Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]). 
    C.shape[0] must be equal to NNN333. 

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb,kc,ld} = 0
    '''
    NNNN3333 = 81 * N**4
    NNN3333 = 81 * N**3

    row = np.arange(NNNN3333)
    col = np.tile(range(NNN3333), N)
    data = np.zeros(NNNN3333)
    data[:] = 1 / sqrt(N)
    c_sum_cplmt = coo_matrix((data, (row, col)), shape=(NNNN3333,NNN3333))

    # bottleneck part
    if mkl:
        C = C.tocsr()
        c_sum_cplmt = c_sum_cplmt.tocsr()

    c_st_cplmt = dot_sp_mats(c_sum_cplmt.transpose(), C, mkl=mkl)
    proj_st_cplmt = dot_sp_mats(c_st_cplmt.transpose(), c_st_cplmt, mkl=mkl)
    # bottleneck part: end
    return proj_st_cplmt

def compressed_complement_projector_sum_rules(C, N, mkl=False):
    '''Return complementary projection matrix for sum rule compressed by C.'''
    return compressed_complement_projector_sum_rules_algo1(C, N, mkl=mkl)

def compressed_projector_sum_rules(C, N, mkl=False):
    '''Return projection matrix for sum rule compressed by C.'''
    proj_cplmt = compressed_complement_projector_sum_rules(C, N, mkl=mkl)
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt

