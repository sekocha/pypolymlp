#!/usr/bin/env python
import argparse
import itertools
import os
import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sympy.physics.wigner import clebsch_gordan

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../c++/lib")
import pymlcpp


def get_lcomb_all(order=2, lproj=0, maxl=9):
    f = open("lists/lcomb/lcomb-o" + str(order) + "-l" + str(lproj))
    l_all = []
    for line in f.readlines():
        lcomb = [
            int(l.replace(",", "").replace("(", "").replace(")", ""))
            for l in line.split()[1:]
        ]
        if max(lcomb) <= maxl:
            l_all.append(lcomb)
    f.close()
    return l_all


def lm_to_matrix_index(l_list, m_list):

    l_plus_m_list = np.array(l_list) + np.array(m_list)
    l_list2 = 2 * np.array(l_list) + 1

    index = 0
    for i, lpm in enumerate(l_plus_m_list):
        index += lpm * np.prod(l_list2[i + 1 :])
    return index


def matrix_index_to_lm(index, l_list):

    n, i_list = index, []
    for l in reversed(l_list):
        i_list.append(n % (2 * l + 1))
        n = n // (2 * l + 1)
    return [i - l for i, l in zip(reversed(i_list), l_list)]


def get_mcomb_all_nonzero(lcomb, lproj=0):
    m_array = [range(-l, l + 1) for l in lcomb]
    m_all = [mcomb for mcomb in itertools.product(*m_array) if abs(sum(mcomb)) <= lproj]
    return m_all


def proj1(lcomb, mcomb1, mcomb2, lproj, mproj):
    l1 = lcomb[0]
    m1, m1p = mcomb1[0], mcomb2[0]
    if l1 == lproj and m1 == m1p == mproj:
        return 1.0
    else:
        return 0.0


def proj2(lcomb, mcomb1, mcomb2, lproj, mproj):
    l1, l2 = lcomb
    (m1, m2), (m1p, m2p) = mcomb1, mcomb2
    val = clebsch_gordan(l1, l2, lproj, m1, m2, mproj) * clebsch_gordan(
        l1, l2, lproj, m1p, m2p, mproj
    )
    return float(val)


def proj3(lcomb, mcomb1, mcomb2, lproj, mproj):
    l1, l2, l3 = lcomb
    (m1, m2, m3), (m1p, m2p, m3p) = mcomb1, mcomb2
    val = 0.0
    for L in range(abs(l1 - l2), l1 + l2 + 1):
        val += (
            clebsch_gordan(l1, l2, L, m1, m2, m1 + m2)
            * clebsch_gordan(l1, l2, L, m1p, m2p, m1p + m2p)
            * clebsch_gordan(l3, L, lproj, m3, m1 + m2, mproj)
            * clebsch_gordan(l3, L, lproj, m3p, m1p + m2p, mproj)
        )
    return float(val)


## Quantum theory of angular momentum (Varshalovich, p.96)
def build_projector(lcomb, mcomb_all, lproj=0, mproj=0):
    size = np.prod(2 * np.array(lcomb) + 1)
    order = len(lcomb)
    if order == 1:
        proj_element = proj1
    elif order == 2:
        proj_element = proj2
    elif order == 3:
        proj_element = proj3

    row, col, data = [], [], []
    iarray = [lm_to_matrix_index(lcomb, mcomb) for mcomb in mcomb_all]
    for c1, c2 in itertools.combinations_with_replacement(range(len(mcomb_all)), 2):
        mcomb1, mcomb2 = mcomb_all[c1], mcomb_all[c2]
        if sum(mcomb1) == sum(mcomb2) == mproj:
            i1, i2 = iarray[c1], iarray[c2]
            val = proj_element(lcomb, mcomb1, mcomb2, lproj, mproj)
            row.append(i1)
            col.append(i2)
            data.append(val)
            if i1 != i2:
                row.append(i2)
                col.append(i1)
                data.append(val)

    return csc_matrix((data, (row, col)), shape=(size, size))


def solve(proj):
    if proj.shape[0] < 501:
        w, v = np.linalg.eig(proj.toarray())
    else:
        w, v = eigsh(proj, k=500)
    col = np.where(w > 0.99)[0]
    return np.real(v[:, col])


def clustering_eigvecs(eigvecs, lcomb):
    lm_uniq, eigvecs1 = [], []
    for n in range(proj.shape[0]):
        mcomb = matrix_index_to_lm(n, lcomb)
        lm = sorted(zip(lcomb, mcomb))
        try:
            index = lm_uniq.index(lm)
            eigvecs1[index] += np.array(eigvecs[n])
        except ValueError:
            eigvecs1.append(np.array(eigvecs[n]))
            lm_uniq.append(lm)

    return np.array(eigvecs1), lm_uniq


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


def print_eigvecs(fname, lcomb, lproj, mproj, eigvecs, lm_uniq):
    f = open(fname, "a")
    for i, eig in enumerate(eigvecs.T):
        nonzero = np.where(np.abs(eig) > 1e-10)[0]
        print(
            " uniq vector: l =",
            lcomb,
            i + 1,
            len(nonzero),
            " l,m(proj) = ",
            lproj,
            mproj,
            file=f,
        )
        for lm, c in zip(np.array(lm_uniq)[nonzero], eig[nonzero]):
            for l, m in lm:
                print(l, m, end=" ", file=f)
            print("%1.15g" % c, file=f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxl", type=int, default=5, help="Max l value.")
    parser.add_argument("--order", type=int, required=True, help="n-th order product.")
    args = parser.parse_args()

    args.lproj = 0

    fname = "./lists/coeffs/coeffs-o" + str(args.order) + "-l" + str(args.lproj)
    if os.path.exists(fname):
        os.remove(fname)

    lcomb_all = get_lcomb_all(order=args.order, lproj=args.lproj, maxl=args.maxl)
    print(lcomb_all)
    # for lcomb in lcomb_all:
    #    mcomb_all = get_mcomb_all_nonzero(lcomb, lproj=args.lproj)
    #    for mproj in range(-args.lproj,args.lproj+1):
    #        print(' l = ', lcomb, 'lp, mp =', args.lproj, mproj)
    #        print(' -- building projector --.')
    #        proj = build_projector\
    #            (lcomb, mcomb_all, lproj=args.lproj, mproj=mproj)
    #        print(' ... done.')
    #        print(' -- solving eigenvalue problem of projector --.')
    #        eigvecs = solve(proj)
    #        eigvecs, lm_uniq = clustering_eigvecs(eigvecs, lcomb)
    #        print(' ... done.')
    #        print(' -- searching independent linear combination (PCA) --.')
    #        eigvecs = uniq_eigvecs(eigvecs)
    #        print(' ... done.')
    #        print_eigvecs(fname, lcomb, args.lproj, mproj, eigvecs, lm_uniq)
