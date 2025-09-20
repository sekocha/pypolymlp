"""Generator of polymlp for disordered model."""

import copy

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

    return params_rand


# def generate_disorder_mlp_pair(
#     params: PolymlpParams,
#     coeffs: np.ndarray,
#     occupancy: tuple,
# ):
#     """Generate MLP with pair invariants for disorder model."""
#     pass


def set_mapping_original_mlp(params: PolymlpParams, coeffs: np.ndarray):
    """Set mapping of original MLP."""
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)
    print(len(atomtype_pair_dict))
    for k, v in atomtype_pair_dict.items():
        print(k, v)

    seq_id = 0
    # gtinv = params.model.gtinv
    if params.model.feature_type == "gtinv":
        for i, attr in enumerate(features_attr):
            # attr: (radial_id, gtinv_id, type_pair_id)
            # print("- id:               ", seq_id)
            # print("  feature_id:       ", i)
            # print("  radial_id:        ", attr[0])
            # print("  gtinv_id:         ", attr[1])
            # print(
            #     "  l_combination:    ",
            #     gtinv.l_comb[attr[1]],
            # )
            # print("  atomtype_pair_ids:", attr[2])

            # print("")
            seq_id += 1


# def generate_disorder_mlp_gtinv(
#    params: PolymlpParams,
#    coeffs: np.ndarray,
#    occupancy: tuple,
# ):
#    """Generate MLP with polynomial invariants for disorder model."""
#
#     gtinv = params.model.gtinv
#        for i, attr in enumerate(features_attr):
#            print("- id:               ", seq_id, file=f)
#            print("  feature_id:       ", i, file=f)
#            print("  radial_id:        ", attr[0], file=f)
#            print("  gtinv_id:         ", attr[1], file=f)
#            print(
#                "  l_combination:    ",
#                gtinv.l_comb[attr[1]],
#                file=f,
#            )
#            print("  atomtype_pair_ids:", attr[2], file=f)
#            print("", file=f)
#            seq_id += 1
#    pass
#


def generate_disorder_mlp(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP for disorder model."""
    params_rand = _generate_disorder_params(params, occupancy)

    #    print(features_attr)

    # feature_mapping = set_mapping_original_mlp(params, coeffs)
    set_mapping_original_mlp(params, coeffs)

    features_attr_r, polynomial_attr_r, atomtype_pair_dict_r = get_features_attr(
        params_rand
    )


#    if params.model.feature_type == "pair":
#        pass
#    elif params.model.feature_type == "gtinv":
#        generate_disorder_mlp_gtinv()
#

#    print("____________")
#    print(features_attr_r)

# print(n_type)


params, coeffs = load_mlp("polymlp.yaml")
occupancy = [[("La", 0.75), ("Te", 0.25)], [("O", 1.0)]]

generate_disorder_mlp(params, coeffs, occupancy)


def save_polymlp_params_yaml(
    params: PolymlpParams,
    filename: str = "polymlp_params.yaml",
):
    """Save feature attributes to yaml file."""
    np.set_printoptions(legacy="1.21")
    f = open(filename, "w")
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)

    elements = np.array(params.elements)
    print("radial_params:", file=f)
    for i, p in enumerate(params.model.pair_params):
        print("- radial_id: ", i, file=f)
        print("  params:    ", list(p), file=f)
        print("", file=f)
    print("", file=f)

    print("atomtype_pairs:", file=f)
    for k, v in atomtype_pair_dict.items():
        print("- pair_id:       ", k, file=f)
        print("  pair_elements: ", file=f)
        for v1 in v:
            print("  - ", list(elements[v1]), file=f)
        print("", file=f)
    print("", file=f)

    print("features:", file=f)
    seq_id = 0
    if params.model.feature_type == "pair":
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  atomtype_pair_ids:", attr[1], file=f)
            print("", file=f)
            seq_id += 1
        print("", file=f)

    elif params.model.feature_type == "gtinv":
        gtinv = params.model.gtinv
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  gtinv_id:         ", attr[1], file=f)
            print(
                "  l_combination:    ",
                gtinv.l_comb[attr[1]],
                file=f,
            )
            print("  atomtype_pair_ids:", attr[2], file=f)
            print("", file=f)
            seq_id += 1
        print("", file=f)

    if len(polynomial_attr) > 0:
        print("polynomial_features:", file=f)
        for i, attr in enumerate(polynomial_attr):
            print("- id:          ", seq_id, file=f)
            print("  feature_ids: ", attr, file=f)
            print("", file=f)
            seq_id += 1

    f.close()
    return seq_id
