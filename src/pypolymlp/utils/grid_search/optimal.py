"""Functions for finding optimal MLPs."""

import os

import numpy as np
import yaml
from scipy.spatial import ConvexHull


def write_yaml(data, system, filename="polymlp_summary_all.yaml"):
    """Save optimal polymlps to a yaml file."""
    f = open(filename, "w")
    print("system:", system, file=f)
    print("", file=f)
    print("unit:", file=f)
    print("  cost:         msec/atom/step", file=f)
    print("  rmse_energy:  meV/atom", file=f)
    print("  rmse_force:   eV/ang.", file=f)

    print("", file=f)
    print("polymlps:", file=f)
    for d in data:
        print("- id:    ", d[-2], file=f)
        print("  path:  ", d[-1], file=f)
        print("  cost_single:", d[0], file=f)
        print("  cost_openmp:", d[1], file=f)
        print("  rmse_energy:", d[2], file=f)
        print("  rmse_force: ", d[3], file=f)
        print("", file=f)

    f.close()


def find_optimal_mlps(dirs, key, use_force=False, use_logscale_time=False):
    """Find optimal polymlps on convex hull."""
    d_array = []
    system = None
    for dir_pot in dirs:
        fname1 = dir_pot + "/polymlp_error.yaml"
        fname2 = dir_pot + "/polymlp_cost.yaml"
        match_d = None
        if os.path.exists(fname1) and os.path.exists(fname2):
            print(dir_pot)
            yml_data = yaml.safe_load(open(fname1))

            if "prediction_errors" in yml_data:
                values = yml_data["prediction_errors"]
            else:
                values = yml_data["prediction_errors_test"]

            for d in values:
                if key in d["dataset"]:
                    match_d = d
                    break

        if match_d is not None:
            abspath = os.path.abspath(dir_pot)
            rmse_e = match_d["rmse_energy"]
            rmse_f = match_d["rmse_force"]
            yml_data = yaml.safe_load(open(fname2))
            time1 = yml_data["costs"]["single_core"]
            time36 = yml_data["costs"]["openmp"]
            name = dir_pot.split("/")[-1]
            if system is None:
                system = yml_data["system"]
            d_array.append([time1, time36, rmse_e, rmse_f, name, abspath])

    d_array = sorted(d_array, key=lambda x: x[0])
    d_array = np.array(d_array)

    write_yaml(d_array, system, filename="polymlp_summary_all.yaml")

    if use_force:
        d_target = d_array[:, [0, 3]]
    else:
        d_target = d_array[:, [0, 2]]

    if use_logscale_time:
        d_target = np.vstack(
            [
                np.log10(d_target[:, 0].astype(float)),
                d_target[:, 1].astype(float),
            ]
        ).T

    ch = ConvexHull(d_target)
    v_convex = np.unique(ch.simplices)

    v_convex_l = []
    for v1 in v_convex:
        lower_convex = True
        time1 = float(d_array[v1][0])
        rmse_e1, rmse_f1 = float(d_array[v1][2]), float(d_array[v1][3])
        for v2 in v_convex:
            if v1 != v2:
                time2 = float(d_array[v2][0])
                rmse_e2, rmse_f2 = float(d_array[v2][2]), float(d_array[v2][3])
                if time2 < time1 and rmse_e2 < rmse_e1 and rmse_f2 < rmse_f1:
                    lower_convex = False
                    break
        if lower_convex:
            v_convex_l.append(v1)

    d_convex = d_array[np.array(v_convex_l)]
    d_convex = d_convex[d_convex[:, 2].astype(float) < 30]

    write_yaml(d_convex, system, filename="polymlp_summary_convex.yaml")
