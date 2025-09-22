"""Generator of polymlp for disordered model."""

import copy
import itertools

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.io_polymlp import load_mlp
from pypolymlp.mlp_dev.core.features_attr import get_features_attr

# from typing import Optional


def _check_params(params: PolymlpParams):
    """Check constraints for parameters."""
    if not params.type_full:
        raise RuntimeError("type_full must be True")

    uniq_ids = set()
    for ids in params.model.pair_params_conditional.values():
        uniq_ids.add(tuple(ids))

    if len(uniq_ids) != 1:
        raise RuntimeError("All pair_params_conditional must be the same.")

    return True


def _generate_disorder_params(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    _check_params(params)

    params_rand = copy.deepcopy(params)
    map_mass = dict(zip(params.elements, params.mass))

    elements_rand, mass_rand = [], []
    type_group = []
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")

        mass, type_tmp = 0.0, []
        for ele, comp in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")

            mass += map_mass[ele] * comp
            type_tmp.append(params.elements.index(ele))

        elements_rand.append(occ[0][0])
        mass_rand.append(mass)
        type_group.append(type_tmp)

    params_rand.n_type = len(occupancy)
    params_rand.elements = elements_rand
    params_rand.mass = mass_rand

    # TODO: Modify type_pairs and type_indices
    # Use type_group

    occupancy_type = [
        [(params.elements.index(ele), comp) for ele, comp in occ] for occ in occupancy
    ]
    return params_rand, occupancy_type


def set_mapping_original_mlp(params: PolymlpParams, coeffs: np.ndarray):
    """Set mapping of original MLP."""
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)

    seq_id = 0
    map_feature = dict()
    if params.model.feature_type == "gtinv":
        gtinv = params.model.gtinv
        features = []
        for attr in features_attr:
            # attr: (radial_id, gtinv_id, type_pair_id)
            rad_id = attr[0]
            lc = gtinv.l_comb[attr[1]]
            tp = [tuple(atomtype_pair_dict[i][0]) for i in attr[2]]
            key = tuple([rad_id, tuple(lc), tuple(tp)])
            features.append(key)
            map_feature[key] = coeffs[seq_id]
            seq_id += 1
    else:
        pass

    map_polynomial = dict()
    if len(polynomial_attr) > 0:
        for attr in polynomial_attr:
            key = tuple([features[i] for i in attr])
            map_polynomial[key] = coeffs[seq_id]
            seq_id += 1

    return map_feature, map_polynomial


def generate_disorder_mlp_pair(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP with pair invariants for disorder model."""
    map_feature, map_polynomial = set_mapping_original_mlp(params, coeffs)
    params_rand, occupancy_type = _generate_disorder_params(params, occupancy)
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params_rand)

    # TODO: function for pair


def find_central_types(type_pairs: list):
    """Find atom types of central atoms for a combination of type pairs."""
    common_type = set(type_pairs[0])
    for tp in type_pairs[1:]:
        common_type = common_type & set(tp)

    central = dict()
    for t1 in common_type:
        neighbors = []
        for tp in type_pairs:
            neigh = tp[1] if tp[0] == t1 else tp[0]
            neighbors.append(neigh)
        central[t1] = neighbors
    return central


def generate_disorder_mlp_gtinv(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP with polynomial invariants for disorder model."""
    map_feature, map_polynomial = set_mapping_original_mlp(params, coeffs)
    params_rand, occupancy_type = _generate_disorder_params(params, occupancy)
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params_rand)

    gtinv = params_rand.model.gtinv
    coeffs_rand = []
    for attr in features_attr:
        rad_id = attr[0]
        lc = tuple(gtinv.l_comb[attr[1]])
        tp = [tuple(atomtype_pair_dict[i][0]) for i in attr[2]]
        central = find_central_types(tp)
        print(lc, tp)

        coeff = 0.0
        weight = 1.0 / len(central)
        print(weight)
        for ctype, neigh_types in central.items():
            print("Neighbor types:")
            print([occupancy_type[i] for i in neigh_types])
            occ_comb = itertools.product(*[occupancy_type[i] for i in neigh_types])
            coeff_tmp = 0.0
            for occ in occ_comb:
                print("Neighbor type combinations:")
                print(occ)
                tp = [tuple(sorted([ctype, t])) for t, p in occ]
                print(tp)
                prob = np.prod([p for t, p in occ])
                print(prob)

                lc_tp = sorted([(l, t) for l, t in zip(lc, tp)])
                tp_key = tuple([t for l, t in lc_tp])

                key = tuple([rad_id, tuple(lc), tp_key])
                coeff_tmp += prob * map_feature[key]
            coeff += weight * coeff_tmp

        coeffs_rand.append(coeff)

    if len(polynomial_attr) > 0:
        pass


def generate_disorder_mlp(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP for disorder model."""
    if params.model.feature_type == "pair":
        generate_disorder_mlp_pair(params, coeffs, occupancy)
    elif params.model.feature_type == "gtinv":
        generate_disorder_mlp_gtinv(params, coeffs, occupancy)


params, coeffs = load_mlp("polymlp.yaml")
occupancy = [[("La", 0.75), ("Te", 0.25)], [("O", 1.0)]]

generate_disorder_mlp(params, coeffs, occupancy)


# def save_polymlp_params_yaml(
#     params: PolymlpParams,
#     filename: str = "polymlp_params.yaml",
# ):
#     """Save feature attributes to yaml file."""
#     print("features:", file=f)
#     seq_id = 0
#     if params.model.feature_type == "pair":
#         for i, attr in enumerate(features_attr):
#             print("- id:               ", seq_id, file=f)
#             print("  feature_id:       ", i, file=f)
#             print("  radial_id:        ", attr[0], file=f)
#             print("  atomtype_pair_ids:", attr[1], file=f)
#             print("", file=f)
#             seq_id += 1
#         print("", file=f)
#
#     elif params.model.feature_type == "gtinv":
#         gtinv = params.model.gtinv
#         for i, attr in enumerate(features_attr):
#             print("- id:               ", seq_id, file=f)
#             print("  feature_id:       ", i, file=f)
#             print("  radial_id:        ", attr[0], file=f)
#             print("  gtinv_id:         ", attr[1], file=f)
#             print(
#                 "  l_combination:    ",
#                 gtinv.l_comb[attr[1]],
#                 file=f,
#             )
#             print("  atomtype_pair_ids:", attr[2], file=f)
#             print("", file=f)
#             seq_id += 1
#         print("", file=f)
#
#     if len(polynomial_attr) > 0:
#         print("polynomial_features:", file=f)
#         for i, attr in enumerate(polynomial_attr):
#             print("- id:          ", seq_id, file=f)
#             print("  feature_ids: ", attr, file=f)
#             print("", file=f)
#             seq_id += 1
#
#     f.close()
#     return seq_id
