#!/usr/bin/env python
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from pypolymlp.calculator.compute_features import compute_from_polymlp_lammps
from pypolymlp.core.interface_vasp import parse_structures_from_poscars

# from lammps_api.go.random.post.structure_matcher.structure_matcher \
#                                                import fit_poscars
#


def __connected_components(x_equivs, tol_distance=1e-0):

    if x_equivs.shape[0] < 300:
        dist = cdist(x_equivs, x_equivs)
        graph = np.array(dist < tol_distance, dtype=int)
        graph = csr_matrix(graph)
        n_components, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )
    else:
        list1 = [i for i in range(x_equivs.shape[0])]
        label_min = 0
        n_split = round(sqrt(len(list1)))
        for ids in np.array_split(list1, n_split):
            x_tmp = x_equivs[ids]
            dist = cdist(x_tmp, x_tmp)
            graph = np.array(dist < tol_distance, dtype=int)
            graph = csr_matrix(graph)
            n_comps, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )
            labels_dict = defaultdict(list)
            for ids, lab in zip(ids, labels):
                labels_dict[lab + label_min].append(ids)
            label_min += n_comps

        ids = [ids[0] for ids in labels_dict.values()]
        x_tmp = x_equivs[ids]
        dist = cdist(x_tmp, x_tmp)
        graph = np.array(dist < tol_distance, dtype=int)
        graph = csr_matrix(graph)
        n_comps, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )

        labels_all = np.zeros(x_equivs.shape[0], dtype=int)
        for i, lab in enumerate(labels):
            for j in labels_dict[i]:
                labels_all[j] = lab

        n_components = len(set(labels_all))
        labels = labels_all

    replace = [min(np.where(labels == i)[0]) for i in set(labels)]
    labels = np.array([replace[l1] for l1 in labels])

    return n_components, labels


def compute_features(pot, summary, coeffs=True, scales=True):

    poscars = [item["id"] for item in summary]
    st_dicts = parse_structures_from_poscars(poscars)
    x, mlp_dict = compute_from_polymlp_lammps(st_dicts, pot=pot, return_mlp_dict=True)
    rec_n_atoms_sums = [1.0 / item["n_atoms_sum"] for item in summary]
    if coeffs is True and scales is True:
        weights = mlp_dict["coeffs"] / mlp_dict["scales"]
    elif coeffs is False and scales is True:
        weights = 1.0 / mlp_dict["scales"]
    elif coeffs is True and scales is False:
        weights = mlp_dict["coeffs"]
    else:
        weights = 1.0

    x = np.multiply(x.T, rec_n_atoms_sums).T
    x = np.multiply(x, weights)
    return x


def get_equivalency(
    summary_values,
    pot=None,
    tol_distance=1e-3,
    tol_energy=1e-4,
    pmg_matcher=False,
    verbose=False,
):
    """
    Parameters
    ----------
    summary_values: dict (key: e, spg, id, n_iter, n_atoms_sum)
    """

    equivalent_class = list(range(len(summary_values)))

    n_target = 10
    for i1, item1 in enumerate(summary_values):
        e1, spg1 = item1["e"], item1["spg"]
        ibegin = max(0, i1 - n_target)
        iend = min(i1 + n_target, len(summary_values))
        target = summary_values[ibegin:iend]
        itarget = list(range(ibegin, iend))

        for i2, item2 in zip(itarget, target):
            e2, spg2 = item2["e"], item2["spg"]
            if (
                i1 != i2
                and abs(e1 - e2) < tol_energy
                and len(set(spg1) & set(spg2)) != 0
            ):
                i3 = min(equivalent_class[i1], equivalent_class[i2])
                equivalent_class[i1] = i3
                equivalent_class[i2] = i3

    if pot is not None:
        x = compute_features(pot, summary_values)
        print("Feature shape =", x.shape)

        print("Calculating equivalent class ...")
        orbits = defaultdict(list)
        for i, eq in enumerate(equivalent_class):
            orbits[eq].append(i)

        for rep, equivs in orbits.items():
            if len(equivs) > 1:
                x_equivs = x[equivs]
                _, labels = __connected_components(x_equivs, tol_distance=tol_distance)
                """
                if pmg_matcher:
                    _, labels3 = __connected_components(x_equivs,
                                                        tol_distance=2e-2)
                    _, labels2 = __connected_components(x_equivs,
                                                        tol_distance=1e-0)

                    print(labels)
                    print(labels3)
                    print(labels2)
                    for i in np.where(labels != labels2)[0]:
                        for j in range(i):
                            match = fit_poscars(summary_values[equivs[i]][2],
                                                summary_values[equivs[j]][2])
                            if match:
                                labels[i] = labels[j]
                                break
                    print(labels)
                """

                n_components = len(set(labels))
                for i, lab in enumerate(labels):
                    equivalent_class[equivs[i]] = equivs[lab]

                if verbose:
                    for eq in equivs:
                        print(
                            eq,
                            summary_values[eq]["id"],
                            summary_values[eq]["e"],
                            summary_values[eq]["spg"],
                        )

                dist = cdist([x_equivs[0]], x_equivs)

                if verbose:
                    print(equivs[0], dist)
                    print(
                        "rep_id =",
                        rep,
                        "n_rep / n_equivs =",
                        n_components,
                        "/",
                        len(equivs),
                    )

    orbits = defaultdict(list)
    for i, eq in enumerate(equivalent_class):
        orbits[eq].append(i)

    return orbits
