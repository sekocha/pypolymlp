#!/usr/bin/env python
import numpy as np
import sys, os, time
from math import *
from joblib import Parallel, delayed
import itertools
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import mlpcpp

from mlptools.common.structure import Structure
from mlptools.common.readvasp import Poscar

def electrostatic_parallel_with_components(st_array, 
                                           charge_array, 
                                           accuracy=1e-12, 
                                           wfactor=0.01, 
                                           force=True,
                                           n_jobs=-1):

    if force == True:
        er, eg, e_self, e, fr, fg, f, sr, sg, s \
            = zip(*Parallel(n_jobs=n_jobs)\
                 (delayed(electrostatic_with_components)(st, charge, 
                                                         accuracy=accuracy, 
                                                         wfactor=wfactor, 
                                                         force=True)
                  for st, charge in zip(st_array, charge_array)))

        f = [f2 for f1 in f for f2 in f1]
        fr = [f2 for f1 in fr for f2 in f1]
        fg = [f2 for f1 in fg for f2 in f1]
        s = [s2 for s1 in s for s2 in s1]
        sr = [f2 for f1 in sr for f2 in f1]
        sg = [f2 for f1 in sg for f2 in f1]
        return np.array(e), np.array(er), np.array(eg), np.array(e_self), \
               np.array(f), np.array(fr), np.array(fg), \
               np.array(s), np.array(sr), np.array(sg)
    else:
        er, eg, e_self, e = zip(*Parallel(n_jobs=n_jobs)
            (delayed(electrostatic_with_components)(st, charge, 
                                                    accuracy=accuracy,
                                                    wfactor=wfactor, 
                                                    force=False) 
            for st, charge in zip(st_array, charge_array)))
        return np.array(e), np.array(er), np.array(eg), np.array(e_self)

def electrostatic_with_components(st, 
                                  charge, 
                                  accuracy=1e-12, 
                                  wfactor=0.01, 
                                  force=True):

    eta, rmax, gmax = params(st, accuracy=accuracy, wfactor=wfactor)
#    print(' rmax, gmax =',  rmax, gmax)
#    if rmax > 6.0:
#        rmax = 6.0
#    print(' rmax, gmax =',  rmax, gmax)
    gvectors = get_gvectors(st, gmax)

    n_type = len(st.get_n_atoms())
    obj = mlpcpp.Ewald(st.get_axis(),
                       st.get_positions_cartesian(),
                       st.get_types(),
                       n_type,
                       rmax,
                       gvectors,
                       charge,
                       st.get_volume(),
                       eta,
                       force)

    if force==False:
        return (obj.get_real_energy(), 
                obj.get_reciprocal_energy(), 
                obj.get_self_energy(), 
                obj.get_energy())
    else:
        return (obj.get_real_energy(),
                obj.get_reciprocal_energy(),
                obj.get_self_energy(), 
                obj.get_energy(),
                np.asarray(obj.get_real_force()), 
                np.asarray(obj.get_reciprocal_force()), 
                np.asarray(obj.get_force()), 
                np.asarray(obj.get_real_stress()),
                np.asarray(obj.get_reciprocal_stress()),
                np.asarray(obj.get_stress()))

def electrostatic_parallel(st_array, 
                           charge_array, 
                           accuracy=1e-12, 
                           wfactor=0.01, 
                           force=True,
                           n_jobs=-1):

    if force == True:
        _, _, _, e, f, s = zip(*Parallel(n_jobs=n_jobs)\
            (delayed(electrostatic)(st, charge, accuracy=accuracy, \
            wfactor=wfactor, force=True) \
            for st, charge in zip(st_array, charge_array)))
        f = [f2 for f1 in f for f2 in f1]
        s = [s2 for s1 in s for s2 in s1]
        return np.array(e), np.array(f), np.array(s)
    else:
        _, _, _, e = zip(*Parallel(n_jobs=n_jobs)\
            (delayed(electrostatic)(st, charge, accuracy=accuracy, \
            wfactor=wfactor, force=False) \
            for st, charge in zip(st_array, charge_array)))
        return np.array(e)

def electrostatic(st, charge, accuracy=1e-18, wfactor=0.01, force=True):

    eta, rmax, gmax = params(st, accuracy=accuracy, wfactor=wfactor)
    gvectors = get_gvectors(st, gmax)

    n_type = len(st.get_n_atoms())
    obj = mlpcpp.Ewald(st.get_axis(),
                       st.get_positions_cartesian(),
                       st.get_types(),
                       n_type,
                       rmax,
                       gvectors,
                       charge,
                       st.get_volume(),
                       eta,
                       force)

    if force==False:
        return obj.get_real_energy(), obj.get_reciprocal_energy(), \
               obj.get_self_energy(), obj.get_energy()
    else:
        return obj.get_real_energy(), obj.get_reciprocal_energy(), \
               obj.get_self_energy(), obj.get_energy(), \
               np.asarray(obj.get_force()), np.asarray(obj.get_stress())


def params(st, accuracy=1e-12, wfactor=0.1):

    n_atom = sum(st.get_n_atoms())
    vol = st.get_volume()

    eta = pow((n_atom * wfactor * pow(pi,3.0) / vol), 0.3333333333)
    rmax = sqrt(-log(accuracy)/eta)
    gmax = 2.0 * sqrt(eta) * sqrt(-log(accuracy))

    return eta, rmax, gmax

def get_gvectors(st, gmax):
    reciprocal_axis = st.get_reciprocal_axis()
    vecs = np.array([[1,0,0],[0,1,0],[0,0,1],
                     [1,1,0],[1,0,1],[0,1,1],
                     [1,1,1],[-1,1,0],[1,-1,0],
                     [-1,0,1],[1,0,-1],[0,-1,1],[0,1,-1]])
    length = [np.linalg.norm(np.dot(reciprocal_axis, v)) for v in vecs]
    expand = int(gmax/min(length)) + 1
    trans_all = itertools.product(*[range(-expand,expand+1) for i in range(3)])
    trans_all = np.array([trans for trans in trans_all]).T
    gvectors = np.dot(reciprocal_axis, trans_all)
    gvec_norms = np.linalg.norm(gvectors, axis=0)
    index = np.where((gvec_norms < gmax) & (gvec_norms > 0))[0]
    return gvectors[:,index].T

def numerical_force_check(st, charge, f):
    eps = 0.0001
    for j in range(st.get_positions().shape[1]):
        for i in range(3):
            st1 = copy.deepcopy(st)
            st1.positions_c[i,j] += eps
            _, _, _, e = electrostatic(st1, 
                                       charge, 
                                       accuracy=1e-12, 
                                       wfactor=0.01, 
                                       force=False)
            print('num, direct', -(e-e0)/eps, f[3*j+i])

def numerical_stress_check(st, charge, s):
    eps = 1e-5
    for i, index in enumerate([[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]]):
        expand = np.identity(3)
        expand[index[0], index[1]] += eps
        axis_new = np.dot(expand, st.get_axis())
        st1 = Structure(axis_new, 
                        st.get_positions(), 
                        st.get_n_atoms(), 
                        st.get_elements(), 
                        st.get_types())
        _, _, _, e = electrostatic(st1, 
                                   charge, 
                                   accuracy=1e-12, 
                                   wfactor=0.01, 
                                   force=False)
        print('num, direct', -(e-e0)/eps, s[i])

if __name__ == '__main__':

    st = Poscar('POSCAR').get_structure_class()
    charge = [2.0,2.0,2.0,2.0,-2.0,-2.0,-2.0,-2.0]
    
    er, eg, eself, e0, f, s = electrostatic(st, 
                                            charge, 
                                            accuracy=1e-12, 
                                            wfactor=0.1, 
                                            force=True)
    
    print(' energy (real) = ', er)
    print(' energy (rec.) = ', eg)
    print(' energy (self) = ', eself)
    print(' energy (all)  = ', e0)
    print(' force  = ')
    print(list(f))
    print(' stress  = ')
    print(list(s))
    
    #numerical_force_check(st, charge, f)
    #numerical_stress_check(st, charge, s)
    
    
