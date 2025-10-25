"""Functions for neighboring atoms."""

import argparse
import signal
from collections import defaultdict

import numpy as np

from pypolymlp.core.interface_vasp import Poscar, Vasprun
from pypolymlp.cxx.lib import libmlpcpp


def find_active_distances(structure, cutoff, decimals=1):

    distances = compute_neighbor_list(structure, cutoff)
    element_map = dict(
        set([(t, ele) for ele, t in zip(structure.elements, structure.types)])
    )

    distances_dict = defaultdict(list)
    for atom1, dis in enumerate(distances):
        type1 = structure.types[atom1]
        for type2, dis2 in enumerate(dis):
            key = tuple(sorted([type1, type2]))
            distances_dict[key].extend(dis2)

    new_dict = dict()
    for k, v in distances_dict.items():
        key = (element_map[k[0]], element_map[k[1]])
        new_dict[key] = np.unique(np.round(v, decimals=decimals))

    distances_dict = new_dict
    # for k, v in distances_dict.items():
    #     print(k, v)

    return distances_dict


def compute_neighbor_list(structure, cutoff):

    if structure.positions_cartesian is None:
        structure.positions_cartesian = structure.axis @ structure.positions

    n_type = len(set(structure.types))
    obj = libmlpcpp.Neighbor(
        structure.axis,
        structure.positions_cartesian,
        structure.types,
        n_type,
        cutoff,
    )
    distances = obj.get_distances()
    return distances


def find_minimum_distance(structure, each_atom=False):

    if structure.positions_cartesian is None:
        structure.positions_cartesian = structure.axis @ structure.positions

    axis_min = np.min(np.linalg.norm(structure.axis, axis=0))
    cutoff = axis_min + 1e-5
    n_type = len(set(structure.types))
    obj = libmlpcpp.Neighbor(
        structure.axis,
        structure.positions_cartesian,
        structure.types,
        n_type,
        cutoff,
    )
    distances = obj.get_distances()
    min_array = [min(d2) for d1 in distances for d2 in d1]
    if each_atom:
        return min_array
    return min(min_array)


def get_coordination_numbers(distances):
    return [len(d2) for d1 in distances for d2 in d1]


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
    parser.add_argument("--cutoff", type=float, default=6.0, help="cutoff radius")
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=None,
        help="cutoff = min(distances) * cutoff_ratio",
    )
    args = parser.parse_args()

    if args.poscars is not None:
        st_array = [Poscar(f).structure for f in args.poscars]
        ids = args.poscars
    elif args.vaspruns is not None:
        st_array = [Vasprun(f).structure for f in args.vaspruns]
        ids = args.vaspruns

    for st, id1 in zip(st_array, ids):
        if args.cutoff_ratio is not None:
            min_dis = find_minimum_distance(st)
            cutoff = min_dis * args.cutoff_ratio
        else:
            cutoff = args.cutoff

        distances = compute_neighbor_list(st, cutoff)
        coord = get_coordination_numbers(distances)
        # print(id1, coord)

        find_active_distances(st, cutoff)
