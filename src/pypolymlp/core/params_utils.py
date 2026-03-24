"""Functions for setting input parameters."""

import itertools
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParamsSingle,
)
from pypolymlp.core.units import HartreetoEV


def set_regression_alphas(alpha_params: tuple):
    """Set regularization parameters in regression."""
    assert len(alpha_params) == 3
    return np.linspace(alpha_params[0], alpha_params[1], round(alpha_params[2]))


def set_element_properties(
    elements: list[str],
    n_type: Optional[int] = None,
    atomic_energy_unit: Literal["eV", "Hartree"] = "eV",
    atomic_energy: Optional[list] = None,
    enable_spins: Optional[tuple] = None,
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

    if atomic_energy_unit == "Hartree":
        atomic_energy = [e * HartreetoEV for e in atomic_energy]

    if enable_spins is not None:
        assert n_type == len(enable_spins)

    return (elements, n_type, atomic_energy, enable_spins)


def set_gtinv_params(
    n_type: int,
    feature_type: Literal["pair", "gtinv"] = "gtinv",
    gtinv_order: int = 3,
    gtinv_maxl: tuple[int] = (4, 4, 2, 1, 1),
    gtinv_version: Literal[1, 2] = 1,
):
    """Set parameters for group-theoretical invariants."""
    if feature_type == "pair":
        empty = PolymlpGtinvParams(order=0, max_l=[], n_type=n_type)
        max_l = 0
        return empty, max_l

    gtinv = PolymlpGtinvParams(
        order=gtinv_order,
        max_l=gtinv_maxl,
        n_type=n_type,
        version=gtinv_version,
    )
    max_l = max(gtinv_maxl)
    return gtinv, max_l


def set_gaussian_params(
    params1: tuple[float, float, int] = (1.0, 1.0, 1),
    params2: tuple[float, float, int] = (0.0, 5.0, 7),
    n_gaussians: Optional[int] = None,
    cutoff: Optional[float] = None,
):
    """Set parameters for Gaussian radial functions."""
    if n_gaussians is None:
        assert len(params1) == len(params2) == 3
        g_params1 = np.linspace(float(params1[0]), float(params1[1]), int(params1[2]))
        g_params2 = np.linspace(float(params2[0]), float(params2[1]), int(params2[2]))
        pair_params = list(itertools.product(g_params1, g_params2))
    else:
        if cutoff is None:
            raise RuntimeError("Cutoff required for automatic setting of Gaussians.")

        rmax = cutoff - 1.0
        g_params2 = np.linspace(0.0, 1.0, n_gaussians - 1) * rmax
        width = max(1.0, g_params2[1] - g_params2[0])
        pair_params = [(width, p2) for p2 in g_params2]

    pair_params.append((0.0, 0.0))
    return pair_params


def set_active_gaussian_params(
    pair_params: np.ndarray,
    elements: list,
    distance: Optional[dict] = None,
):
    """Set parameters for active Gaussian radial functions."""
    if distance is not None:
        for ele_pair, dis_array in distance.items():
            if len(ele_pair) != 2:
                raise RuntimeError("Keys of distance must be element pair.")
            if not isinstance(dis_array, (list, tuple, np.ndarray)):
                raise RuntimeError("Values of distance must be array-type.")

        cond = True
        distance_replace = dict()
        for ele_pair in distance.keys():
            key = tuple(sorted(ele_pair, key=lambda x: elements.index(x)))
            distance_replace[key] = distance[ele_pair]
        distance = distance_replace
    else:
        cond = False
        distance = dict()

    atomtypes = dict()
    for i, ele in enumerate(elements):
        atomtypes[ele] = i

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


def set_all_params(
    elements: tuple[str] = None,
    include_force: bool = True,
    include_stress: bool = False,
    cutoff: float = 6.0,
    model_type: Literal[1, 2, 3, 4] = 4,
    max_p: Literal[1, 2, 3] = 2,
    feature_type: Literal["pair", "gtinv"] = "gtinv",
    gaussian_params1: tuple[float, float, int] = (1.0, 1.0, 1),
    gaussian_params2: tuple[float, float, int] = (0.0, 5.0, 7),
    n_gaussians: Optional[int] = None,
    distance: Optional[dict] = None,
    reg_alpha_params: tuple[float, float, int] = (-3.0, 1.0, 5),
    gtinv_order: int = 3,
    gtinv_maxl: tuple[int] = (4, 4, 2, 1, 1),
    gtinv_version: Literal[1, 2] = 1,
    atomic_energy_unit: Literal["eV", "Hartree"] = "eV",
    atomic_energy: Optional[tuple[float]] = None,
    enable_spins: Optional[tuple] = None,
    rearrange_by_elements: bool = True,
):
    """Assign input parameters.

    Parameters
    ----------
    elements: Element species, (e.g., ['Mg','O'])
    include_force: Considering force entries
    include_stress: Considering stress entries
    cutoff: Cutoff radius
    model_type: Polynomial function type
        model_type = 1: Linear polynomial of polynomial invariants
        model_type = 2: Polynomial of polynomial invariants
        model_type = 3: Polynomial of pair invariants
                        + linear polynomial of polynomial invariants
        model_type = 4: Polynomial of pair and second-order invariants
                        + linear polynomial of polynomial invariants
    max_p: Order of polynomial function
    feature_type: 'gtinv' or 'pair'
    gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
        Parameters are given as np.linspace(p[0], p[1], p[2]),
        where p[0], p[1], and p[2] are given by gaussian_params1
        and gaussian_params2.
    n_gaussians: Number of Gaussian functions.
                 Parameters of Gaussians are automatically determined
                 by the cutoff radius and number of Gaussians.
    distance: Interatomic distances for element pairs.
        (e.g.) distance = {(Sr, Sr): [3.5, 4.8], (Ti, Ti): [2.5, 5.5]}
    reg_alpha_params: Parameters for penalty term in
        linear ridge regression. Parameters are given as
        np.linspace(p[0], p[1], p[2]).
    gtinv_order: Maximum order of polynomial invariants.
    gtinv_maxl: Maximum angular numbers of polynomial invariants.
        [maxl for order=2, maxl for order=3, ...]
    atomic_energy: Atomic energies.
    enable_spins: Boolean array to activate spin configuration.
    rearrange_by_elements: Set True if not developing special MLPs.
    """
    elements, n_type, atomic_energy, enable_spins = set_element_properties(
        elements,
        n_type=len(elements),
        atomic_energy_unit=atomic_energy_unit,
        atomic_energy=atomic_energy,
        enable_spins=enable_spins,
    )
    element_order = elements if rearrange_by_elements else None
    alphas = set_regression_alphas(reg_alpha_params)

    gtinv, max_l = set_gtinv_params(
        n_type,
        feature_type=feature_type,
        gtinv_order=gtinv_order,
        gtinv_maxl=gtinv_maxl,
        gtinv_version=gtinv_version,
    )
    pair_params = set_gaussian_params(
        gaussian_params1,
        gaussian_params2,
        n_gaussians=n_gaussians,
        cutoff=cutoff,
    )
    pair_params_active, pair_cond = set_active_gaussian_params(
        pair_params,
        elements,
        distance,
    )

    model = PolymlpModelParams(
        cutoff=cutoff,
        model_type=model_type,
        max_p=max_p,
        max_l=max_l,
        feature_type=feature_type,
        gtinv=gtinv,
        pair_type="gaussian",
        pair_conditional=pair_cond,
        pair_params=pair_params,
        pair_params_conditional=pair_params_active,
    )
    params = PolymlpParamsSingle(
        n_type=n_type,
        elements=elements,
        model=model,
        atomic_energy=atomic_energy,
        enable_spins=enable_spins,
        regression_alpha=alphas,
        include_force=include_force,
        include_stress=include_stress,
        element_order=element_order,
    )
    return params
