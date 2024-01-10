#!/usr/bin/env python
import numpy as np
import glob
import itertools
import time 
import scipy.integrate

from joblib import Parallel, delayed

from mlptools.common.readvasp import Poscar
from mlptools.common.structure import Structure

from basis import radial_basis
from basis import gaussian_atomic_density
from basis import Basis
from basis import BasisSet

def prediction_atomic_density(expansion_coeffs, basis_array, rgrid):

    pred_array = []
    for r_in in rgrid:
        pred = 0.0
        for c, basis in zip(expansion_coeffs, basis_array):
            pred += c * basis.eval(r_in)
        pred_array.append(pred)
    return np.array(pred_array)


def parallel_func(poscar, basis_array, ub, rgrid):

    st = Poscar(poscar).get_structure_class()
    dis, _ ,_ = st.calc_get_neighbors(rcut)
    rmse_array = []
    for d in dis:
        r_st = d[0]
        exp_coeffs = basis_set.compute_expansion_coeffs(r_st, lb=0, ub=ub)
        true = np.array([gaussian_atomic_density(r, r_st) for r in rgrid])
        pred = prediction_atomic_density(exp_coeffs, basis_array, rgrid)
        rmse = np.sqrt(np.mean(np.square(pred - true)))
        rmse_array.append(rmse)
    return rmse_array

if __name__ == '__main__':

    rcut = 6.0

    #beta_array = np.linspace(1.0, 10, 2)
    #r0_array = np.linspace(0.0, rcut, 5)

    beta_array = np.linspace(2, 2, 1)
    r0_array = np.linspace(0.0, rcut, 20)
    params = itertools.product(beta_array, r0_array)

    basis_array = [Basis(radial_basis, args=(beta, r0, rcut)) 
                   for beta, r0 in params]
    basis_set = BasisSet(basis_array)
    basis_set.compute_overlap(lb=0.0, ub=rcut)

    rmin, ngrid = 1.0, 100
    rgrid = np.linspace(rmin, rcut-0.5, num=ngrid)

    ######
    # define a structure
    #r_st = list(np.linspace(1.0, rcut, num=200))

    dir_p = '/home/seko/home-nas0/mlip-dft/1-unary/2019-n10000/Al/finished/'
    poscars = sorted(glob.glob(dir_p + '*/POSCAR'))[:1000]

    if len(poscars) < 36:
        n_jobs = len(poscars)
    else:
        n_jobs = -1

    t1 = time.time()
    rmse_all = Parallel(n_jobs=n_jobs)(delayed(parallel_func)
                            (p, basis_array, rcut, rgrid) for p in poscars)
    t2 = time.time()
    print(t2-t1)

    print(' average rmse =', np.mean([r2 for r1 in rmse_all for r2 in r1]))


