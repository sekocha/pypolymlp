"""Functions for finding optimal MLPs."""

import os

import numpy as np
import yaml
from scipy.spatial import ConvexHull


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
    d_array, system = _parse_errors_costs(dirs, key, verbose=verbose)
    _write_yaml(d_array, system, filename=filename_all)

    d_convex = _find_convex_mlps(
        d_array,
        use_force=use_force,
        use_logscale_time=use_logscale_time,
    )
    _write_yaml(d_convex, system, filename=filename_convex)
    return d_array, d_convex, system


def _find_convex_mlps(
    d_array: np.ndarray,
    use_force: bool = False,
    use_logscale_time: bool = False,
    rmse_threshold: float = 30,
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

    v_convex_l = _find_lower_convex(data_values)
    d_convex = np.hstack([data_values[v_convex_l], data_str[v_convex_l]])
    d_convex = d_convex[d_convex[:, 2].astype(float) < rmse_threshold]
    return d_convex


def _find_lower_convex(data_values: np.ndarray):
    """Find only data on lower convex hull."""
    v_convex_l = []
    for i1, d1 in enumerate(data_values):
        lower_convex = not any(
            i1 != i2 and d2[0] < d1[0] and d2[2] < d1[2] and d2[3] < d1[3]
            for i2, d2 in enumerate(data_values)
        )
        if lower_convex:
            v_convex_l.append(i1)
    v_convex_l = np.array(v_convex_l)
    return v_convex_l


def _parse_errors(filename: str, key: str):
    """Parse errors from files."""
    if not os.path.exists(filename):
        return None

    yml_data = yaml.safe_load(open(filename))
    if "prediction_errors" in yml_data:
        values = yml_data["prediction_errors"]
    elif "prediction_errors_test" in yml_data:
        values = yml_data["prediction_errors_test"]
    else:
        return None

    match_d = None
    for d in values:
        if key in d["dataset"]:
            match_d = d
            break

    if match_d is None:
        return None

    rmse_e = match_d["rmse_energy"]
    rmse_f = match_d["rmse_force"]
    return rmse_e, rmse_f


def _parse_costs(filename: str):
    """Parse costs from files."""
    if not os.path.exists(filename):
        return None

    yml_data = yaml.safe_load(open(filename))
    time1 = yml_data["costs"]["single_core"]
    time_mp = yml_data["costs"]["openmp"]

    try:
        system = yml_data["system"]
    except:
        system = None
    return time1, time_mp, system


def _parse_errors_costs(dirs: list, key: str, verbose: bool = False):
    """Parse errors and costs from files."""
    d_array = []
    for dir_pot in dirs:
        if verbose:
            print("Parsing", dir_pot, flush=True)

        fname1 = dir_pot + "/polymlp_error.yaml"
        res = _parse_errors(fname1, key)
        if res is None:
            continue
        rmse_e, rmse_f = res

        fname2 = dir_pot + "/polymlp_cost.yaml"
        res = _parse_costs(fname2)
        if res is None:
            continue
        time1, time_mp, system = res
        name = dir_pot.split("/")[-1]
        abspath = os.path.abspath(dir_pot)
        d_array.append([time1, time_mp, rmse_e, rmse_f, name, abspath])

    d_array = np.array(sorted(d_array, key=lambda x: x[0]))
    return d_array, system


def _write_yaml(
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
