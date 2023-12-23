#!/usr/bin/env python
import numpy as np
import itertools
from math import sqrt
import time

import scipy
from scipy.sparse import csr_matrix, coo_matrix

from pypolymlp.symfc_dev.linalg import dot_sp_mats

def compressed_complement_projector_sum_rules_algo1(C, N, mkl=False):
    '''Return complementary projection matrix for sum rule compressed by C.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]). 
    C.shape[0] must be equal to NN33. 

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb} = 0
    '''
    NN33 = 9 * N**2
    N33 = 9 * N

    row = np.arange(NN33)
    col = np.tile(range(N33), N)
    data = np.zeros(NN33)
    data[:] = 1 / sqrt(N)
    c_sum_cplmt = coo_matrix((data, (row, col)), shape=(NN33,N33))

    if mkl:
        C = C.tocsr()
        c_sum_cplmt = c_sum_cplmt.tocsr()

    # bottleneck part
    c_sum_cplmt_compr = dot_sp_mats(c_sum_cplmt.transpose(), C, mkl=mkl)
    proj_sum_cplmt = dot_sp_mats(c_sum_cplmt_compr.transpose(), 
                                c_sum_cplmt_compr, mkl=mkl)
    # bottleneck part: end
    return proj_sum_cplmt

def compressed_complement_projector_sum_rules(C, N, mkl=False):
    '''Return complementary projection matrix for sum rule compressed by C.'''
    return compressed_complement_projector_sum_rules_algo1(C, N, mkl=mkl)

def compressed_projector_sum_rules(C, N, mkl=False):
    '''Return projection matrix for sum rule compressed by C.'''
    proj_cplmt = compressed_complement_projector_sum_rules(C, N, mkl=mkl)
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


