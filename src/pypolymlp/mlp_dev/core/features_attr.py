"""Functions for obtaining feature attributes."""

from collections import defaultdict

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.cxx.lib import libmlpcpp


def get_num_features(params: PolymlpParams):
    """Return number of features."""
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)
    n_fearures = len(features_attr) + len(polynomial_attr)
    return n_fearures


def get_features_attr(params: PolymlpParams, element_swap: bool = False):
    """Get feature attributes."""
    params.element_swap = element_swap
    obj = libmlpcpp.FeaturesAttr(params.as_dict())

    type_pairs = obj.get_type_pairs()
    atomtype_pair_dict = defaultdict(list)
    for type1, tps in enumerate(type_pairs):
        for type2, tp in enumerate(tps):
            atomtype_pair_dict[tp].append([type1, type2])

    radial_ids = obj.get_radial_ids()
    tcomb_ids = obj.get_tcomb_ids()
    polynomial_attr = obj.get_polynomial_ids()
    if params.model.feature_type == "pair":
        features_attr = list(zip(radial_ids, tcomb_ids))
    elif params.model.feature_type == "gtinv":
        gtinv_ids = obj.get_gtinv_ids()
        features_attr = list(zip(radial_ids, gtinv_ids, tcomb_ids))

    return features_attr, polynomial_attr, atomtype_pair_dict


def write_polymlp_params_yaml(
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
