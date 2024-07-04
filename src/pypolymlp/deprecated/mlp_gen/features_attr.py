#!/usr/bin/env python
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.cxx.lib import libmlpcpp


def get_features_attr(params_dict, element_swap=False):

    params_dict["element_swap"] = element_swap
    obj = libmlpcpp.FeaturesAttr(params_dict)

    type_comb_pair = obj.get_type_comb_pair()
    atomtype_pair_dict = defaultdict(list)
    for i, tc_pair in enumerate(type_comb_pair):
        for type1, type2s in enumerate(tc_pair):
            for type2 in type2s:
                atomtype_pair_dict[i].append([type1, type2])

    radial_ids = obj.get_radial_ids()
    tcomb_ids = obj.get_tcomb_ids()
    polynomial_attr = obj.get_polynomial_ids()
    if params_dict["model"]["feature_type"] == "pair":
        features_attr = list(zip(radial_ids, tcomb_ids))
    elif params_dict["model"]["feature_type"] == "gtinv":
        gtinv_ids = obj.get_gtinv_ids()
        features_attr = list(zip(radial_ids, gtinv_ids, tcomb_ids))

    return features_attr, polynomial_attr, atomtype_pair_dict


def write_polymlp_params_yaml(params_dict, filename="polymlp_params.yaml"):

    f = open(filename, "w")

    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params_dict)

    elements = np.array(params_dict["elements"])
    print("radial_params:", file=f)
    for i, p in enumerate(params_dict["model"]["pair_params"]):
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
    if params_dict["model"]["feature_type"] == "pair":
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  atomtype_pair_ids:", attr[1], file=f)
            print("", file=f)
            seq_id += 1
        print("", file=f)

    elif params_dict["model"]["feature_type"] == "gtinv":
        gtinv_dict = params_dict["model"]["gtinv"]
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  gtinv_id:         ", attr[1], file=f)
            print(
                "  l_combination:    ",
                gtinv_dict["l_comb"][attr[1]],
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        default="polymlp.in",
        help="Input file name",
    )
    args = parser.parse_args()

    params_dict = ParamsParser(args.infile).get_params()
    write_polymlp_params_yaml(params_dict)
