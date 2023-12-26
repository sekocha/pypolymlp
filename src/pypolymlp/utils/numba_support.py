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

