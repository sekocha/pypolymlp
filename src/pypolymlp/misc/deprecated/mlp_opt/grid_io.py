#!/usr/bin/env python
import itertools
import os


def write_params_dict(params_dict, filename):

    f = open(filename, "w")

    print("feature_type", params_dict["feature_type"], file=f)
    print("cutoff", params_dict["cutoff"], file=f)

    gauss1, gauss2 = params_dict["gauss1"], params_dict["gauss2"]
    print("gaussian_params1", gauss1[0], gauss1[1], gauss1[2], file=f)
    print("gaussian_params2", gauss2[0], gauss2[1], gauss2[2], file=f)

    reg = params_dict["reg_alpha_params"]
    print("reg_alpha_params", reg[0], reg[1], reg[2], file=f)
    print("", file=f)

    print("model_type", params_dict["model_type"], file=f)
    print("max_p", params_dict["max_p"], file=f)

    if params_dict["feature_type"] == "gtinv":
        print("gtinv_order", params_dict["gtinv_order"], file=f)
        print("gtinv_maxl", end="", file=f)
        for maxl in params_dict["gtinv_maxl"]:
            print("", maxl, end="", file=f)
        print("", file=f)

    print("", file=f)

    print("include_force", params_dict["include_force"], file=f)
    print("include_stress", params_dict["include_stress"], file=f)

    f.close()


def write_grid(params_grid, iseq=0):

    for i, params_dict in enumerate(params_grid):
        idx = str(iseq + 1).zfill(5)
        dirname = "model_grid/polymlp-" + idx + "/"
        os.makedirs(dirname, exist_ok=True)
        write_params_dict(params_dict, dirname + "polymlp.in")
        iseq += 1

    return iseq


def write_grid_hybrid(params_grid1, params_grid2, iseq=0):

    grid_pairs = itertools.product(params_grid1, params_grid2)
    for params_dict1, params_dict2 in grid_pairs:
        idx = str(iseq + 1).zfill(5)
        dirname = "model_grid/polymlp-hybrid-" + idx + "/"
        os.makedirs(dirname, exist_ok=True)
        write_params_dict(params_dict1, dirname + "polymlp.in")
        write_params_dict(params_dict2, dirname + "polymlp.in.2")
        iseq += 1

    return iseq
