"""Functions for finding optimal MLPs."""

import os

import numpy as np
import yaml
from scipy.spatial import ConvexHull


def write_yaml(
    data: np.ndarray,
    system: str,
    filename: str = "polymlp_summary_all.yaml",
):
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


def find_convex_mlps(
    d_array: np.ndarray,
    use_force: bool = False,
    use_logscale_time: bool = False,
):
    """Find convex polymlps from data."""
    d_target = d_array[:, [0, 3]] if use_force else d_array[:, [0, 2]]
    if use_logscale_time:
        time = np.log10(d_target[:, 0].astype(float))
        error = d_target[:, 1].astype(float)
        d_target = np.vstack([time, error]).T

    ch = ConvexHull(d_target)
    v_convex = np.unique(ch.simplices)

    data_values = d_array[v_convex, :4].astype(float)
    data_str = d_array[v_convex, 4:]

    v_convex_l = []
    for i1, d1 in enumerate(data_values):
        lower_convex = True
        time1, rmse_e1, rmse_f1 = d1[0], d1[2], d1[3]
        for i2, d2 in enumerate(data_values):
            if i1 == i2:
                continue
            time2, rmse_e2, rmse_f2 = d2[0], d2[2], d2[3]
            if time2 < time1 and rmse_e2 < rmse_e1 and rmse_f2 < rmse_f1:
                lower_convex = False
                break
        if lower_convex:
            v_convex_l.append(i1)
    v_convex_l = np.array(v_convex_l)
    d_convex = np.hstack([data_values[v_convex_l], data_str[v_convex_l]])
    d_convex = d_convex[d_convex[:, 2].astype(float) < 30]
    return d_convex


def parse_errors_costs(dirs: list, key: str, verbose: bool = False):
    """Parse errors and costs from files."""
    d_array = []
    system = None
    for dir_pot in dirs:
        fname1 = dir_pot + "/polymlp_error.yaml"
        if not os.path.exists(fname1):
            continue

        fname2 = dir_pot + "/polymlp_cost.yaml"
        if not os.path.exists(fname2):
            continue

        if verbose:
            print("Parsing", dir_pot, flush=True)
        yml_data = yaml.safe_load(open(fname1))
        if "prediction_errors" in yml_data:
            values = yml_data["prediction_errors"]
        elif "prediction_errors_test" in yml_data:
            values = yml_data["prediction_errors_test"]
        else:
            continue

        match_d = None
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

    d_array = np.array(sorted(d_array, key=lambda x: x[0]))
    return d_array, system


def find_optimal_mlps(
    dirs: list,
    key: str,
    use_force: bool = False,
    use_logscale_time: bool = False,
    filename_all: str = "polymlp_summary_all.yaml",
    filename_convex: str = "polymlp_summary_convex.yaml",
    verbose: bool = False,
):
    """Find optimal polymlps on convex hull."""
    d_array, system = parse_errors_costs(dirs, key, verbose=verbose)
    write_yaml(d_array, system, filename=filename_all)

    d_convex = find_convex_mlps(
        d_array,
        use_force=use_force,
        use_logscale_time=use_logscale_time,
    )
    write_yaml(d_convex, system, filename=filename_convex)
    return d_array, d_convex, system
