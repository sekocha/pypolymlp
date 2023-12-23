#!/usr/bin/env python
import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

def solve_linear_equation(A, b):
    """
    numpy and scipy implementations
    x = np.linalg.solve(A, b)
    x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
    """
    posv, = get_lapack_funcs(('posv',), (A, b))
    _, x, _ = posv(A, b, lower=False,
                   overwrite_a=False,
                   overwrite_b=False)
    return x

def solve_linear_ridge_equation(A, b, alpha=1e-5):
    """
    numpy and scipy implementations
    x = np.linalg.solve(A, b)
    x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
    """
    A += alpha * np.eye(A.shape[0])
    posv, = get_lapack_funcs(('posv',), (A, b))
    _, x, _ = posv(A, b, lower=False,
                   overwrite_a=False,
                   overwrite_b=False)
    return x

def fit(X, y):
    '''
    Solving a normal equation in least-squares 
    (X.T @ X) @ coefs = X.T @ y
    '''
    n_samples, n_features = X.shape
    A = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    coefs = solve_linear_equation(A, Xy)
    return coefs

