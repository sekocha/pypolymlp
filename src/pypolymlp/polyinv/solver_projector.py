#!/usr/bin/env python
import numpy as np
import sys, os
import argparse
import itertools

from scipy.sparse import csr_matrix

#from scipy.sparse.linalg import eigsh
#from sklearn.decomposition import PCA
#

from pypolymlp.polyinv.enumerate_lcombs import n_all
from pypolymlp.polyinv.cxx.lib import libprojcpp


def build_projector(lcomb, mcomb_all):

    '''Quantum theory of angular momentum (Varshalovich, p.96)'''
    size = np.prod(2 * np.array(lcomb) + 1)
    obj = libprojcpp.Projector()
    obj.build_projector(lcomb, mcomb_all);

    data, row, col = obj.get_data(), obj.get_row(), obj.get_col()
    return csr_matrix((data, (row, col)), shape=(size, size))


def get_mcomb_all_nonzero(lcomb, lproj=0):

    if (lproj == 0):
        m_array = []
        for l in lcomb[:-1]:
            m_array.append(range(-l,l+1))
            m_array.append(range(-l,l+1))
        m_all = list(itertools.product(*m_array))
    else:
        m_all = []
        '''
        m_array = [range(-l,l+1) for l in lcomb]
        m_all = [mcomb for mcomb in itertools.product(*m_array) 
                 if abs(sum(mcomb)) <= lproj]
        '''
    return m_all


def lm_to_matrix_index(l_list, m_list):
    
    l_plus_m_list = np.array(l_list) + np.array(m_list)
    l_list2 = 2*np.array(l_list)+1

    index = 0
    for i, lpm in enumerate(l_plus_m_list):
        index += lpm * np.prod(l_list2[i+1:])
    return index

def matrix_index_to_lm(index, l_list):

    n, i_list = index, []
    for l in reversed(l_list):
        i_list.append(n % (2*l+1))
        n = n // (2*l+1)
    return [i - l for i, l in zip(reversed(i_list), l_list)]

def solve(proj):
    if (proj.shape[0] < 501):
        w, v = np.linalg.eig(proj.toarray())
    else:
        w, v = eigsh(proj, k=500)
    col = np.where(w > 0.99)[0]
    return np.real(v[:,col])


def uniq_eigvecs(eigvecs):
    tol = 1e-10
    if np.all(np.abs(eigvecs) < tol):
        return np.array([])
    elif eigvecs.shape[1] == 1:
        return eigvecs
    else:
        pca = PCA(n_components=eigvecs.shape[1]).fit(eigvecs.T)
        uniq_index = np.where(pca.explained_variance_ratio_ > tol)[0]
        return pca.components_[uniq_index].T

def print_eigvecs(fname, lcomb, lproj, mproj, eigvecs, lm_uniq, minl=11):
    if max(lcomb) >= minl:
        f = open(fname, 'a')
        for i, eig in enumerate(eigvecs.T):
            nonzero = np.where(np.abs(eig) > 1e-10)[0]
            print(' independent invariant: l =', lcomb, i+1, \
                len(nonzero), ' l,m(proj) = ', lproj, mproj, file=f)
            for lm, c in zip(np.array(lm_uniq)[nonzero], eig[nonzero]):
                for l, m in lm:
                    print(l, m, end=' ', file=f)
                print("%1.15g" % c, file=f)
        f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--maxl', type=int, default=5, help='Max l value.'
    )
    parser.add_argument(
        '--order', type=int, default=None, help='n-th order product.'
    )
    args = parser.parse_args()

    args.lproj = 0
    orders = list(range(2,3)) if args.order is None else [args.order]
    for order in orders:
        n_list, lcomb_all = n_all(args.maxl, order, args.lproj)
        for lcomb in lcomb_all:
            mcomb_all = get_mcomb_all_nonzero(lcomb, lproj=args.lproj)
            for mproj in range(-args.lproj, args.lproj + 1):
                print('lcomb:', lcomb)
                print('- lp:', args.lproj)
                print('  mp:', mproj)
                print(' -- building projector --.')
                proj = build_projector(lcomb, mcomb_all)
                print(proj)



#    for lcomb in lcomb_all:
#        mcomb_all = get_mcomb_all_nonzero(lcomb, lproj=args.lproj)
#        for mproj in range(-args.lproj,args.lproj+1):
#            print(' l = ', lcomb, 'lp, mp =', args.lproj, mproj)
#            print(' -- building projector --.')
#            proj = build_projector(lcomb, mcomb_all)
#            print(' ... done.')
#
#            print(' -- solving eigenvalue problem of projector --.')
#            eigvecs = solve(proj)
##            eigvecs, lm_uniq = clustering_eigvecs(eigvecs, lcomb)
#            print(' ... done.')
#            print(' -- searching independent linear combination (PCA) --.')
#            eigvecs = uniq_eigvecs(eigvecs)
#            print(' ... done.')
#            lm_uniq = [list(zip(lcomb, matrix_index_to_lm(i, lcomb))) \
#                for i in range(proj.shape[0])]
#            print_eigvecs(fname, lcomb, args.lproj, mproj, eigvecs, lm_uniq)
##

#def proj1(lcomb, mcomb1, mcomb2, lproj, mproj):
#    l1 = lcomb[0]
#    m1, m1p = mcomb1[0], mcomb2[0]
#    if (l1 == lproj and m1 == m1p == mproj):
#        return 1.0
#    else:
#        return 0.0
#
#def proj2(lcomb, mcomb1, mcomb2, lproj, mproj):
#    l1, l2 = lcomb
#    (m1,m2), (m1p,m2p) = mcomb1, mcomb2
#    val = clebsch_gordan(l1,l2,lproj,m1,m2,mproj)\
#        *clebsch_gordan(l1,l2,lproj,m1p,m2p,mproj)
#    return float(val)
#
#def proj3(lcomb, mcomb1, mcomb2, lproj, mproj):
#    l1, l2, l3 = lcomb
#    (m1,m2,m3), (m1p,m2p,m3p) = mcomb1, mcomb2
#    val = 0.0
#    for L in range(abs(l1-l2),l1+l2+1):
#        val += clebsch_gordan(l1,l2,L,m1,m2,m1+m2)\
#            *clebsch_gordan(l1,l2,L,m1p,m2p,m1p+m2p)\
#            *clebsch_gordan(l3,L,lproj,m3,m1+m2,mproj)\
#            *clebsch_gordan(l3,L,lproj,m3p,m1p+m2p,mproj)
#    return float(val)


