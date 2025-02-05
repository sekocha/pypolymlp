#!/usr/bin/env python
import argparse
import os
import shutil

import yaml

from pypolymlp.mlp_opt.grid_io import write_params_dict


def set_models(
    cutoffs,
    stress,
    reg_alpha_params,
    feature_type="gtinv",
    gauss1=[0.5, 0.5, 1],
    r_gauss2=2.0,
    n_gauss2=2,
    model_type=4,
    max_p=2,
    gtinv_maxl=[12, 8],
):

    params_dict_all = []
    for i, cut in enumerate(cutoffs):
        params_dict = dict()
        params_dict["feature_type"] = feature_type
        params_dict["cutoff"] = cut
        params_dict["gauss1"] = gauss1
        params_dict["gauss2"] = [0.0, cut - r_gauss2, n_gauss2]

        params_dict["reg_alpha_params"] = reg_alpha_params
        params_dict["model_type"] = model_type
        params_dict["max_p"] = max_p

        if feature_type == "gtinv":
            params_dict["gtinv_maxl"] = gtinv_maxl
            params_dict["gtinv_order"] = len(params_dict["gtinv_maxl"]) + 1

        params_dict["include_force"] = True
        params_dict["include_stress"] = stress

        params_dict_all.append(params_dict)

    return params_dict_all


if __name__ == "__main__":

    ps = argparse.ArgumentParser()
    ps.add_argument(
        "--yaml",
        type=str,
        default="polymlp_summary_convex.yaml",
        help="Summary yaml file",
    )
    ps.add_argument("--no_stress", action="store_false", help="Stress")
    args = ps.parse_args()

    f = open(args.yaml)
    yamldata = yaml.safe_load(f)
    f.close()

    cutoffs = [4.0, 5.0]
    reg_alpha_params = [-4.0, 3.0, 15]

    grid1 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=2,
        model_type=2,
        max_p=2,
        gtinv_maxl=[12, 8],
    )
    grid2 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=2,
        model_type=2,
        max_p=2,
        gtinv_maxl=[12, 4, 2],
    )
    grid3 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=2,
        model_type=4,
        max_p=2,
        gtinv_maxl=[12, 12, 4, 1, 1],
    )
    params_grid_hybrid_high = grid1
    params_grid_hybrid_high.extend(grid2)
    params_grid_hybrid_high.extend(grid3)

    grid4 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=4,
        model_type=2,
        max_p=2,
        gtinv_maxl=[12, 12],
    )
    grid5 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=2,
        model_type=4,
        max_p=2,
        gtinv_maxl=[12, 12, 2],
    )
    grid6 = set_models(
        cutoffs,
        args.no_stress,
        reg_alpha_params,
        n_gauss2=2,
        model_type=4,
        max_p=3,
        gtinv_maxl=[12, 8],
    )

    params_grid_hybrid_low = grid4
    params_grid_hybrid_low.extend(grid5)
    params_grid_hybrid_low.extend(grid6)

    polymlps = yamldata["polymlps"]
    for pot in polymlps:
        f = open(pot["path"] + "/polymlp.in")
        lines = f.readlines()
        f.close()

        addlines = []
        for line1 in lines:
            if "n_type" in line1:
                addlines.append(line1)
            if "elements" in line1:
                addlines.append(line1)

        if pot["cost_single"] > 2:
            params_grid_hybrid = params_grid_hybrid_high
        else:
            params_grid_hybrid = params_grid_hybrid_low

        for i, params in enumerate(params_grid_hybrid):
            path_output = pot["id"] + "-hybrid-" + str(i + 1).zfill(4)
            os.makedirs(path_output, exist_ok=True)
            shutil.copy(pot["path"] + "/polymlp.in", path_output)
            write_params_dict(params, path_output + "/polymlp.in.2")
            f = open(path_output + "/polymlp.in.2", "a")
            print("", file=f)
            for line1 in addlines:
                print(line1, file=f, end="")
            print("", file=f)
            f.close()
