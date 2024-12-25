"""Functions for setting input parameters."""

import itertools
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpGtinvParams


def set_regression_alphas(alpha_params: tuple):
    """Set regularization parameters in regression."""
    assert len(alpha_params) == 3
    return np.linspace(alpha_params[0], alpha_params[1], round(alpha_params[2]))


def set_element_properties(
    elements: list[str],
    n_type: Optional[int] = None,
    atomic_energy: Optional[list] = None,
):
    """Set properties for identifying elements."""
    if n_type is None:
        n_type = len(elements)
    else:
        assert n_type == len(elements)

    if atomic_energy is None:
        atomic_energy = tuple([0.0 for i in range(n_type)])
    else:
        assert len(atomic_energy) == n_type

    return (elements, n_type, atomic_energy)


def set_gtinv_params(
    n_type: int,
    feature_type: Literal["pair", "gtinv"] = "gtinv",
    gtinv_order: int = 3,
    gtinv_maxl: tuple[int] = (4, 4, 2, 1, 1),
    gtinv_version: Literal[1, 2] = 1,
):
    """Set parameters for group-theoretical invariants."""
    if feature_type == "gtinv":
        gtinv = PolymlpGtinvParams(
            order=gtinv_order,
            max_l=gtinv_maxl,
            n_type=n_type,
            version=gtinv_version,
        )
        max_l = max(gtinv_maxl)
    else:
        gtinv = PolymlpGtinvParams(
            order=0,
            max_l=[],
            n_type=n_type,
        )
        max_l = 0
    return gtinv, max_l


def set_gaussian_params(
    params1: tuple[float, float, int] = (1.0, 1.0, 1),
    params2: tuple[float, float, int] = (0.0, 5.0, 7),
):
    """Set parameters for Gaussian radial functions."""
    assert len(params1) == len(params2) == 3
    g_params1 = np.linspace(float(params1[0]), float(params1[1]), int(params1[2]))
    g_params2 = np.linspace(float(params2[0]), float(params2[1]), int(params2[2]))
    pair_params = list(itertools.product(g_params1, g_params2))
    pair_params.append((0.0, 0.0))
    return pair_params


def set_active_gaussian_params(
    pair_params: np.ndarray,
    elements: list,
    distance: Optional[dict] = None,
):
    """Set parameters for active Gaussian radial functions."""
    atomtypes = dict()
    for i, ele in enumerate(elements):
        atomtypes[ele] = i

    if distance is None:
        cond = False
        distance = dict()
    else:
        cond = True
        for k in distance.keys():
            k = sorted(k, key=lambda x: elements.index(x))

    element_pairs = itertools.combinations_with_replacement(elements, 2)
    pair_params_indices = dict()
    for ele_pair in element_pairs:
        key = (atomtypes[ele_pair[0]], atomtypes[ele_pair[1]])
        if ele_pair not in distance:
            pair_params_indices[key] = list(range(len(pair_params)))
        else:
            match = [len(pair_params) - 1]
            for dis in distance[ele_pair]:
                for i, p in enumerate(pair_params[:-1]):
                    if dis < p[1] + 1 / p[0] and dis > p[1] - 1 / p[0]:
                        match.append(i)
            pair_params_indices[key] = sorted(set(match))

    return pair_params_indices, cond
