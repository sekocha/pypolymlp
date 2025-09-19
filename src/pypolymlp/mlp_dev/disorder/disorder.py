"""Generator of polymlp for disordered model."""

import copy

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.io_polymlp import load_mlp
from pypolymlp.mlp_dev.core.features_attr import get_features_attr

# from typing import Optional


def _check_params(params: PolymlpParams):
    """Check constraints for parameters."""


def _generate_disorder_params(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    params_rand = copy.deepcopy(params)
    map_mass = dict(zip(params.elements, params.mass))
    print(map_mass)
    #   itype = 0
    elements_rand, mass_rand = [], []
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")
        for ele, _ in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")

        mass = 0.0
        for ele, comp in occ:
            mass += map_mass[ele] * comp
        mass_rand.append(mass)
        elements_rand.append(occ[0][0])

    params_rand.n_type = len(occupancy)
    params_rand.elements = elements_rand
    params_rand.mass = mass_rand

    # TODO: Modify type_pairs
    # TODO: Modify type_indices

    return params_rand


def generate_disorder_mlp(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP for disorder model."""
    params_rand = _generate_disorder_params(params, occupancy)

    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)
    print(features_attr)

    features_attr_r, polynomial_attr_r, atomtype_pair_dict_r = get_features_attr(
        params_rand
    )
    print("____________")
    print(features_attr_r)

    # print(n_type)


params, coeffs = load_mlp("polymlp.yaml")
occupancy = [[("La", 0.75), ("Te", 0.25)], [("O", 1.0)]]

generate_disorder_mlp(params, coeffs, occupancy)
