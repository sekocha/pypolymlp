#!/usr/bin/env python
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def mat_prod_vec(mat1, vec1, axis=0):
    if axis == 0:
        # prod(mat[row] * vec1[i])
        for i in prange(mat1.shape[0]):
            val1 = vec1[i]
            for j in prange(mat1.shape[1]):
                mat1[i,j] *= val1
    elif axis == 1:
        # prod(mat[col] * vec1[i])
        for i in prange(mat1.shape[1]):
            val1 = vec1[i]
            for j in prange(mat1.shape[0]):
                mat1[j,i] *= val1
"""

#@njit(parallel=True, fastmath=True)
#def mat_prod(mat1, val1):
#    for i in prange(mat1.shape[1]):
#        mat1[:,i] *= val1
#
#@njit(parallel=True, fastmath=True)
#def mat_divide(mat1, val1):
#    for i in prange(mat1.shape[1]):
#        mat1[:,i] /= val1
#
## prod(mat[col] * vec1[i])
#@njit(parallel=True, fastmath=True)
#def mat_prod_vec(mat1, vec1):
#    for i in prange(mat1.shape[1]):
#        val1 = vec1[i]
#        for j in prange(mat1.shape[0]):
#            mat1[j,i] *= val1
#
## prod(mat[row] * vec1[i])
#@njit(parallel=True, fastmath=True)
#def mat_prod_vec2(mat1, vec1):
#    for i in prange(mat1.shape[0]):
#        val1 = vec1[i]
#        for j in prange(mat1.shape[1]):
#            mat1[i,j] *= val1
#
##@njit(parallel=True, fastmath=True)
##def mat_divide_vec(mat1, vec1):
##    for i in prange(mat1.shape[1]):
##        val1 = vec1[i]
##        for j in prange(mat1.shape[0]):
##            mat1[j,i] /= val1
#
#@njit(parallel=True, fastmath=True)
#def compute_apply_scales(X):
#    scale = np.zeros(X.shape[1])
#    for i in prange(X.shape[1]):
#        scale[i] = np.std(X[:,i])
#        X[:,i] /= scale[i]
#
#    return scale

#@njit(parallel=True, fastmath=True)
#def vec_divide_vec(vec1, vec2):
#    for i in prange(vec1.shape[0]):
#        vec1[i] /= vec2[i]
#    return vec1
#
#@njit(parallel=True, fastmath=True)
#def mat_dot_vec(mat1, vec1):
#    array = np.zeros(mat1.shape[0])
#    for i in prange(mat1.shape[0]):
#        array[i] = np.dot(mat1[i], vec1)
#    return array
#
#@njit(parallel=True, fastmath=True)
#def mat_add(mat1, mat2):
#    for i in prange(mat1.shape[0]):
#        mat1[i] += mat2[i]
#    return mat1
#
"""#
