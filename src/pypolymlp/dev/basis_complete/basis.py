#!/usr/bin/env python
import numpy as np
import itertools
from math import *

import scipy.integrate

def gaussian_atomic_density(r, r_array, beta_st=20.0):
    return sum([gaussian(r, beta_st, r1) for r1 in r_array])

def gaussian(r, beta, r0):
    coeff = sqrt(beta) / sqrt(pi)
    return coeff * exp(- beta * pow((r - r0), 2))

def cosine_cutoff(r, rcut):
    if r < rcut:
        return 0.5 * (cos(pi * r / rcut) + 1)
    return 0.0

def radial_basis(r, *args):
    return gaussian(r, args[0], args[1]) * cosine_cutoff(r, args[2])

class Basis:

    def __init__(self, func, coeff=None, args=None):

        self.func = func
        if coeff is not None:
            self.coeff = coeff
        else:
            self.coeff = 1.0
        self.args = args

    def set_coeff(self, coeff):
        self.coeff = coeff

    def eval(self, x):
        return self.coeff * self.func(x, *self.args)

class BasisSet:

    def __init__(self, basis_array):
        self.basis_array = basis_array
        self.n_basis = len(basis_array)

    def compute_overlap(self, lb=0, ub=np.inf):

        self.S = np.zeros((self.n_basis,self.n_basis))
        for i, j in itertools.combinations_with_replacement\
                                                (range(self.n_basis),2):
            b1, b2 = self.basis_array[i], self.basis_array[j]

            def func_prod(r):
                return b1.eval(r) * b2.eval(r)

            self.S[i,j] = scipy.integrate.quad(func_prod, lb, ub)[0]

        A_array = np.reciprocal(np.sqrt(np.diag(self.S)))
        self.S = self.get_symmetric(self.S)
        self.S *= np.outer(A_array, A_array)
        self.Sinv = np.linalg.inv(self.S)

        for a, b in zip(A_array, self.basis_array):
            b.set_coeff(a)

    def compute_expansion_coeffs(self, r_st, lb=0, ub=np.inf):

        inner_prod = []
        for basis in self.basis_array:
            def func1(r):
                return gaussian_atomic_density(r, r_st) * basis.eval(r)
            integral = scipy.integrate.quad(func1, lb, ub)[0]
            inner_prod.append(integral)

        return np.dot(np.array(inner_prod), self.Sinv) 

    def get_symmetric(self, a):
        a = np.triu(a)
        return a + a.T - np.diag(a.diagonal())


