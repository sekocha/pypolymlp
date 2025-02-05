#!/usr/bin/env python
import argparse
import itertools
import os
import shutil

import numpy as np
import yaml

from pypolymlp.core.parser_polymlp_params import ParamsParser
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


def model_for_pair(stress, reg_alpha_params, params_dict_in):

    cutoff = params_dict_in["model"]["cutoff"]
    cutoffs = [cutoff - 1.0, cutoff - 2.0]

    maxl_cands = [[4], [8], [4, 4]]
    params_dict_all = []
    for ml, mp in itertools.product(maxl_cands, [2, 3]):
        grid1 = set_models(
            cutoffs,
            stress,
            reg_alpha_params,
            n_gauss2=3,
            model_type=4,
            max_p=mp,
            gtinv_maxl=ml,
        )
        params_dict_all.extend(grid1)

    return params_dict_all


def model_for_gtinv(stress, reg_alpha_params, params_dict_in):

    cutoff = params_dict_in["model"]["cutoff"]
    maxl = params_dict_in["model"]["gtinv"]["max_l"]

    cutoffs = np.arange(4.0, cutoff - 0.999, 1.0)

    maxl_cands = []
    if len(maxl) == 1:
        maxl_cands.append([min(maxl[0] + 2, 12)])
        maxl_cands.append([min(maxl[0] + 4, 12)])
        maxl_cands.append([min(maxl[0] + 4, 12), min(maxl[0] + 4, 8)])
    elif len(maxl) == 2:
        maxl_cands.append([min(maxl[0] + 2, 12), maxl[1]])
        maxl_cands.append([min(maxl[0] + 4, 12), min(maxl[1] + 2, 12)])
        maxl_cands.append([min(maxl[0] + 4, 12), min(maxl[1] + 2, 12), 2])
    elif len(maxl) == 3:
        maxl_cands.append([min(maxl[0] + 2, 12), maxl[1], maxl[2]])
        maxl_cands.append([min(maxl[0] + 4, 12), min(maxl[1] + 2, 12), maxl[2]])
        maxl_cands.append([min(maxl[0] + 4, 12), min(maxl[1] + 2, 12), maxl[2] + 2])
    else:
        maxl_cands.append([min(l + 4, 12) if i < 2 else l for i, l in enumerate(maxl)])

    params_dict_all = []
    for ml in maxl_cands:
        grid1 = set_models(
            cutoffs,
            stress,
            reg_alpha_params,
            n_gauss2=5,
            model_type=4,
            max_p=2,
            gtinv_maxl=ml,
        )
        params_dict_all.extend(grid1)

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

    reg_alpha_params = [-4.0, 3.0, 15]
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

        params = ParamsParser(
            pot["path"] + "/polymlp.in", parse_vasprun_locations=False
        )
        params_dict = params.get_params()

        if params_dict["model"]["feature_type"] == "gtinv":
            grid = model_for_gtinv(args.no_stress, reg_alpha_params, params_dict)
        else:
            grid = model_for_pair(args.no_stress, reg_alpha_params, params_dict)

        for i, params in enumerate(grid):
            path_output = pot["id"] + "-hybrid-" + str(i + 1).zfill(4)
            os.makedirs(path_output, exist_ok=True)
            shutil.copy(pot["path"] + "/polymlp.in", path_output)
            write_params_dict(params, path_output + "/polymlp.in.2")
            f = open(path_output + "/polymlp.in.2", "a")
            print("", file=f)
            for line1 in addlines:
                print(line1, file=f, end="")
            print("", file=f)
