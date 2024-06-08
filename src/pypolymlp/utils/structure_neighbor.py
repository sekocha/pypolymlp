#!/usr/bin/env python
import argparse
import itertools
import signal

import numpy as np
import numpy.matlib
from scipy.spatial.distance import cdist

from pypolymlp.core.interface_vasp import Poscar, Vasprun


def __find_trans(axis, cutoff):

    # neighbor.cpp should be used.
    # Expansion size should be determined automatically
    m = 7
    ranges = [range(-m, m + 1), range(-m, m + 1), range(-m, m + 1)]
    trans = np.array(list(itertools.product(*ranges))).T
    norm_array = np.linalg.norm(np.dot(axis, trans), axis=0)
    ids = norm_array < cutoff + 1e-10

    trans2 = set(
        [
            tuple(c + np.array(tr))
            for c in trans[:, ids].T
            for tr in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        ]
    )

    trans2 = np.array([[t[0], t[1], t[2]] for t in trans2]).T
    trans2_c = np.dot(axis, trans2)

    return trans2_c.T


def compute_neighbor_list(st_dict, cutoff):

    if "positions_c" not in st_dict:
        positions_c = np.dot(st_dict["axis"], st_dict["positions"])
        st_dict["positions_c"] = positions_c
    else:
        positions_c = st_dict["positions_c"]

    natom, ntype = st_dict["positions"].shape[1], len(st_dict["n_atoms"])

    neigh_dict = dict()
    neigh_dict["distances"] = [[[] for j in range(ntype)] for i in range(natom)]
    neigh_dict["diff_vecs"] = [[[] for j in range(ntype)] for i in range(natom)]
    neigh_dict["atom_ids"] = [[[] for j in range(ntype)] for i in range(natom)]
    neigh_dict["elements"] = [[[] for j in range(ntype)] for i in range(natom)]
    trans_c = __find_trans(st_dict["axis"], cutoff)

    for tr in trans_c:
        tr_mat = np.matlib.repmat(tr.reshape(3, 1), 1, natom)
        dis = cdist(positions_c.T, (positions_c + tr_mat).T)
        index = np.where((dis < cutoff) & (dis > 1e-10))
        for i1, i2 in zip(index[0], index[1]):
            t = st_dict["types"][i2]
            vec = positions_c.T[i2] + tr - positions_c.T[i1]
            neigh_dict["distances"][i1][t].append(dis[i1, i2])
            neigh_dict["diff_vecs"][i1][t].append(vec)
            neigh_dict["atom_ids"][i1][t].append(int(i2))
            neigh_dict["elements"][i1][t].append(st_dict["elements"][i2])

    return neigh_dict


def find_minimum_distance(st_dict, each_atom=False):

    axis_min = np.min(np.linalg.norm(st_dict["axis"], axis=0))
    neigh_dict = compute_neighbor_list(st_dict, axis_min + 1e-5)

    min_array = [min(d2) for d1 in neigh_dict["distances"] for d2 in d1]
    if each_atom:
        return min_array
    return min(min_array)


def get_coordination_numbers(neigh_dict):
    return [len(d2) for d1 in neigh_dict["distances"] for d2 in d1]


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscars",
        nargs="*",
        type=str,
        default=None,
        help="poscar files",
    )
    parser.add_argument(
        "-v",
        "--vaspruns",
        nargs="*",
        type=str,
        default=None,
        help="vasprun files",
    )
    parser.add_argument("--cutoff", type=float, default=None, help="cutoff radius")
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=None,
        help="cutoff = min(distances) * cutoff_ratio",
    )
    args = parser.parse_args()

    if args.poscars is not None:
        st_dict_array = [Poscar(f).get_structure() for f in args.poscars]
        ids = args.poscars
    elif args.vaspruns is not None:
        st_dict_array = [Vasprun(f).get_structure() for f in args.vaspruns]
        ids = args.vaspruns

    for st_dict, id1 in zip(st_dict_array, ids):
        if args.cutoff_ratio is not None:
            min_dis = find_minimum_distance(st_dict)
            neigh_dict = compute_neighbor_list(st_dict, min_dis * args.cutoff_ratio)
        else:
            if args.cutoff is None:
                args.cutoff = 6.0
            neigh_dict = compute_neighbor_list(st_dict, args.cutoff)

        coord = get_coordination_numbers(neigh_dict)
        print(id1, coord)
        # for i, d1 in enumerate(neigh_dict['distances']):
        #    for d2 in d1:
        #        print('  ', i, d2)
