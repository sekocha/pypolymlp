#!/usr/bin/env python
import argparse
import itertools
import os
import signal

import numpy as np
from scipy.sparse import csr_matrix
from symfc.utils.eig_tools import eigsh_projector

from pypolymlp.polyinv.cxx.lib import libprojcpp
from pypolymlp.polyinv.enumerate_lcombs import n_all


def build_projector(lcomb, mcomb_all):
    """Quantum theory of angular momentum (Varshalovich, p.96)"""
    size = np.prod(2 * np.array(lcomb) + 1)
    obj = libprojcpp.Projector()
    obj.build_projector(lcomb, mcomb_all)

    data, row, col = obj.get_data(), obj.get_row(), obj.get_col()
    return csr_matrix((data, (row, col)), shape=(size, size))


def get_mcomb_all_nonzero(lcomb, lproj=0):

    if lproj == 0:
        m_array = []
        for l1 in lcomb[:-1]:
            m_array.append(range(-l1, l1 + 1))
            m_array.append(range(-l1, l1 + 1))
        m_all = list(itertools.product(*m_array))
    else:
        m_all = []
        """
        m_array = [range(-l,l+1) for l in lcomb]
        m_all = [mcomb for mcomb in itertools.product(*m_array)
                 if abs(sum(mcomb)) <= lproj]
        """
    return m_all


def lm_to_matrix_index(l_list, m_list):

    l_plus_m_list = np.array(l_list) + np.array(m_list)
    l_list2 = 2 * np.array(l_list) + 1

    index = 0
    for i, lpm in enumerate(l_plus_m_list):
        index += lpm * np.prod(l_list2[i + 1 :])
    return index


def matrix_index_to_lm(index, l_list):

    n, i_list = index, []
    for l1 in reversed(l_list):
        i_list.append(n % (2 * l1 + 1))
        n = n // (2 * l1 + 1)
    i_list = reversed(i_list)
    return np.array([(l, i - l) for i, l in zip(i_list, l_list)])


def print_eigvecs(lcomb, lproj, mproj, eigvecs, lm_indices, tol=1e-10, file=None):

    for i, eig in enumerate(eigvecs.T):
        print("- lcomb:", list(lcomb), file=file)
        print("  id:   ", i + 1, file=file)
        print("  lproj:", lproj, file=file)
        print("  mproj:", mproj, file=file)

        nonzero = np.where(np.abs(eig) > tol)[0]
        mcombs = [[m for l1, m in lm] for lm in np.array(lm_indices)[nonzero]]
        print("  mcombs_coeffs:", file=file)
        for mcomb, c in zip(mcombs, eig[nonzero]):
            print("  -", [list(mcomb), c], file=file)
        print("", file=f)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minl",
        type=int,
        default=None,
        help="Exclude l combinations composed of only l < minl.",
    )
    parser.add_argument("--maxl", type=int, default=10, help="Maximum l value.")
    parser.add_argument("--order", type=int, default=None, help="n-th order product.")
    args = parser.parse_args()

    os.makedirs("lists_ver2", exist_ok=True)
    args.lproj = 0
    orders = list(range(1, 7)) if args.order is None else [args.order]
    for order in orders:
        n_list, lcomb_all = n_all(args.maxl, order, args.lproj)

        if args.minl is not None:
            lcomb_all = [
                lcomb for lcomb in lcomb_all if np.any(np.array(lcomb) >= args.minl)
            ]

        fname = "lists_ver2/basis-order" + str(order) + "-l" + str(args.lproj) + ".yaml"
        f = open(fname, "w")
        print("basis_set:", file=f)
        for lcomb in lcomb_all:
            mcomb_all = get_mcomb_all_nonzero(lcomb, lproj=args.lproj)
            for mproj in range(-args.lproj, args.lproj + 1):
                print("lcomb:", lcomb)
                print("- lp_mp:", [args.lproj, mproj])
                print("Building projector.", end=" ")
                proj = build_projector(lcomb, mcomb_all)
                print("... Done.")
                print("Solving projector.", end=" ")
                eigvecs = eigsh_projector(proj).toarray()
                print("... Done.")

                lm_indices = [
                    matrix_index_to_lm(i, lcomb) for i in range(proj.shape[0])
                ]
                print_eigvecs(lcomb, args.lproj, mproj, eigvecs, lm_indices, file=f)
        f.close()
