#!/usr/bin/env python
import numpy as np
import sys, os
import argparse
import itertools
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

sys.path.append\
    (os.path.dirname(os.path.abspath(__file__)) + '/../../c++/lib')
import pymlcpp

def read_l_list(filename, sym=False, max_l=None):
    
    f = open(filename)
    lines = f.readlines()
    f.close()

    l_list_array = []
    for line in lines:
        string = line.replace('(','').replace(')','').replace(',','')
        l_list = [int(l) for l in reversed(string.split()[1:])]
        add = 1
        if (max_l != None and len(np.where(np.array(l_list)>max_l)[0]) > 0):
            add = 0
        if (sym==True and len(list(set(l_list))) > 1):
            add = 0
        if (add == 1):
            l_list_array.append(l_list)

    return l_list_array


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

def get_mlist(l_list):
    output = []
    if (len(l_list) == 2):
        l1 = l_list[0]
        mlist = list(itertools.product(range(-l1,l1+1), range(-l1,l1+1)))
    elif (len(l_list) == 3):
        l1, l2 = l_list[0], l_list[1]
        mlist = list(itertools.product(range(-l1,l1+1), range(-l1,l1+1), \
            range(-l2,l2+1), range(-l2,l2+1)))
    elif (len(l_list) == 4):
        l1, l2, l3 = l_list[0], l_list[1], l_list[2]
        mlist = list(itertools.product(range(-l1,l1+1), range(-l1,l1+1), \
            range(-l2,l2+1), range(-l2,l2+1), \
            range(-l3,l3+1), range(-l3,l3+1)))
    elif (len(l_list) == 5):
        l1, l2, l3, l4 = l_list[0], l_list[1], l_list[2], l_list[3]
        mlist = list(itertools.product(range(-l1,l1+1), range(-l1,l1+1), \
            range(-l2,l2+1), range(-l2,l2+1), \
            range(-l3,l3+1), range(-l3,l3+1), \
            range(-l3,l3+1), range(-l4,l4+1)))
    elif (len(l_list) == 6):
        l1, l2, l3, l4, l5 = \
            l_list[0], l_list[1], l_list[2], l_list[3], l_list[4]
        mlist = list(itertools.product(range(-l1,l1+1), range(-l1,l1+1), \
            range(-l2,l2+1), range(-l2,l2+1), \
            range(-l3,l3+1), range(-l3,l3+1), \
            range(-l4,l4+1), range(-l4,l4+1), \
            range(-l5,l5+1), range(-l5,l5+1)))
    return mlist

# Quantum theory of angular momentum (Varshalovich, p.96)
def build_projector(l_list):

    mlist = get_mlist(l_list)

    size = np.prod(2*np.array(l_list)+1)
    obj = pymlcpp.Projector()
    obj.build_projector(l_list, mlist);
    proj = csc_matrix((obj.get_data(), (obj.get_row(), obj.get_col())), \
        shape=(size, size))

    if (proj.shape[0] < 501):
        return proj.toarray()
    else:
        return proj

def invariant_lc_products(l_list, proj):

    if (proj.shape[0] < 501):
        w, v = np.linalg.eig(proj)
    else:
        w, v = eigsh(proj, k=500)

    col = np.where(w > 0.99)

    lm_independent_array, index_array = [], []
    for n in range(0, proj.shape[0]):
        m_list1 = matrix_index_to_lm(n, l_list)
        lm_list1 = sorted(list(zip(l_list, m_list1)), reverse=True)
        if (lm_list1 in lm_independent_array):
            index_array.append(lm_independent_array.index(lm_list1))
        else:
            index_array.append(len(lm_independent_array))
            lm_independent_array.append(lm_list1)

    coeff_array = []
    for i, eigvec in enumerate(np.real(v[:,col[0]]).T):
        coeff_sum = [0.0 for lm in lm_independent_array]
        for n, eig in enumerate(eigvec):
            coeff_sum[index_array[n]] += eig
        coeff_array.append(coeff_sum)

    return lm_independent_array, coeff_array

def independent_lc_products(coeff_array):

    tol = 1e-10
    if (np.linalg.norm(coeff_array) < tol):
        return []
    elif (len(coeff_array) > 1):
        pca = PCA(n_components=len(coeff_array))
        pca.fit(np.array(coeff_array))
        independent_index = np.where(pca.explained_variance_ratio_ > tol)[0]
        return pca.components_[independent_index]

    else:
        return coeff_array

def inverse_symmetry(lm_array, coeff):

    lm_new, coeff_new = [], []
    for lm, c in zip(lm_array, coeff):
        l_list, m_list1 = [a[0] for a in lm], [a[1] for a in lm]
        m_list2 = [-m for m in m_list1]

        lm_list1 = lm
        lm_list2 = sorted(list(zip(l_list, m_list2)), reverse=True)
        if (lm_list2 in lm_new):
            index = lm_new.index(lm_list2)
            coeff_new[index][0] += c
            coeff_new[index][1] -= c
        else:
            lm_new.append(lm_list1)
            coeff_new.append([c, c])
    return lm_new, coeff_new


#l_list = [3, 3, 2]
#proj = build_projector(l_list)
#lm_independent, coeff = invariant_lc_products(l_list, proj)
#coeff = independent_lc_products(coeff)
#
#for i, coeff1 in enumerate(coeff):
#    nonzero = np.where(np.abs(coeff1) > 1e-10)[0]
#    lm_new, coeff_new = [], []
#    for index in nonzero:
#        lm_new.append(lm_independent[index]) 
#        coeff_new.append(coeff1[index])
#    lm_new, coeff_new = inverse_symmetry(lm_new, coeff_new)
#    print(lm_new)
#    print(coeff_new)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxl', type=int, help='Max l value.')
    parser.add_argument('--order', type=int, required=True, \
        help='n-th order product.')
    parser.add_argument('--sym', action='store_true', \
        help='Only symmetric invariants are considered.')
    args = parser.parse_args()

    l_list_array = read_l_list('./lists/l_comb/l_list' + str(args.order), \
        sym=args.sym, max_l=args.maxl)

    for l_list in l_list_array:
        if (args.sym==True):
            f = open('./lists/invariant_test/sym_order' + str(args.order), 'a')
        else:
            f = open('./lists/invariant_test/order' + str(args.order), 'a')

        print(l_list)
        print(' -- building projector --.')
        proj = build_projector(l_list)
        print(' ... done.')

        print(' -- solving eigenvalue problem of projector --.')
        lm_independent, coeff = invariant_lc_products(l_list, proj)
        print(' ... done.')
        print(' -- searching independent linear combination (PCA) --.')
        coeff_independent = independent_lc_products(coeff)
        print(' ... done.')

        print(' n_invariants = ', len(coeff), l_list, file=f)
        print(' n_independent_invariants = ', \
            len(coeff_independent), l_list, file=f)

        for i, coeff1 in enumerate(coeff_independent):
            nonzero = np.where(np.abs(coeff1) > 1e-10)[0]
            lm_new, coeff_new = [], []
            for index in nonzero:
                lm_new.append(lm_independent[index]) 
                coeff_new.append(coeff1[index])
            lm_new, coeff_new = inverse_symmetry(lm_new, coeff_new)
            print(' independent invariant: l =', l_list, i+1, \
                len(coeff_new), file=f)
            for lm, c in zip(lm_new, coeff_new):
                if (abs(c[0]) < 1e-15):
                    c[0] = 0.0
                if (abs(c[1]) < 1e-15):
                    c[1] = 0.0
                for l, m in lm:
                    print(l,m, end=' ', file=f)
                print("%1.15g" % c[0], "%1.15g" % c[1], file=f)
        f.close()
#
