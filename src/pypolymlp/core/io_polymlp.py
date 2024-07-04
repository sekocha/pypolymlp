#!/usr/bin/env python
import numpy as np
from setuptools._distutils.util import strtobool

from pypolymlp.core.utils import mass_table
from pypolymlp.cxx.lib import libmlpcpp


def print_param(dict1, key, fstream, prefix=""):
    print(str(dict1[key]), "#", prefix + key, file=fstream)


def print_array1d(array, fstream, comment="", fmt=None):

    for obj in array:
        if fmt is not None:
            print(fmt.format(obj), end=" ", file=fstream)
        else:
            print(obj, end=" ", file=fstream)
    print("#", comment, file=fstream)


def save_multiple_mlp_lammps(
    multiple_params_dicts, cumulative_n_features, coeffs, scales
):

    multiple_coeffs = []
    multiple_scales = []
    for i, params_dict in enumerate(multiple_params_dicts):
        if i == 0:
            begin, end = 0, cumulative_n_features[0]
        else:
            begin, end = (
                cumulative_n_features[i - 1],
                cumulative_n_features[i],
            )

        save_mlp_lammps(
            params_dict,
            coeffs[begin:end],
            scales[begin:end],
            filename="polymlp.lammps." + str(i + 1),
        )
        multiple_coeffs.append(coeffs[begin:end])
        multiple_scales.append(scales[begin:end])

    return multiple_coeffs, multiple_scales


def save_mlp_lammps(params_dict, coeffs, scales, filename="polymlp.lammps"):

    f = open(filename, "w")
    print_array1d(params_dict["elements"], f, comment="elements")
    model_dict = params_dict["model"]
    print_param(model_dict, "cutoff", f)
    print_param(model_dict, "pair_type", f)
    print_param(model_dict, "feature_type", f)
    print_param(model_dict, "model_type", f)
    print_param(model_dict, "max_p", f)
    print_param(model_dict, "max_l", f)

    if model_dict["feature_type"] == "gtinv":
        gtinv_dict = model_dict["gtinv"]
        print_param(gtinv_dict, "order", f, prefix="gtinv_")
        print_array1d(gtinv_dict["max_l"], f, comment="gtinv_max_l")
        gtinv_sym = [0 for _ in gtinv_dict["max_l"]]
        print_array1d(gtinv_sym, f, comment="gtinv_sym")

    print(len(coeffs), "# n_coeffs", file=f)
    print_array1d(coeffs, f, comment="reg. coeffs", fmt="{0:15.15e}")
    print_array1d(scales, f, comment="scales", fmt="{0:15.15e}")

    print(len(model_dict["pair_params"]), "# n_params", file=f)
    for obj in model_dict["pair_params"]:
        print(
            "{0:15.15f}".format(obj[0]),
            "{0:15.15f}".format(obj[1]),
            "# pair func. params",
            file=f,
        )

    mass = [mass_table()[ele] for ele in params_dict["elements"]]
    print_array1d(mass, f, comment="atomic mass", fmt="{0:15.15e}")
    print("False # electrostatic", file=f)

    if model_dict["feature_type"] == "gtinv":
        print_param(gtinv_dict, "version", f, prefix="gtinv_")

    f.close()


def __read_var(line, dtype=int, return_list=False):

    list1 = line.split("#")[0].split()
    if return_list:
        return [dtype(v) for v in list1]
    return dtype(list1[0])


def load_mlp_lammps(filename="polymlp.lammps"):
    """
    params_dict, mlp_dict = load_mlp_lammps(filename='mlp.lammps')
    """

    mlp_dict = dict()
    params_dict = dict()
    params_dict["model"] = model_dict = dict()
    params_dict["model"]["gtinv"] = gtinv_dict = dict()

    f = open(filename)
    lines = f.readlines()
    f.close()

    idx = 0
    params_dict["elements"] = __read_var(lines[idx], str, return_list=True)
    params_dict["element_order"] = params_dict["elements"]
    params_dict["n_type"] = len(params_dict["elements"])
    idx += 1

    model_dict["cutoff"] = __read_var(lines[idx], float)
    idx += 1
    model_dict["pair_type"] = __read_var(lines[idx], str)
    idx += 1
    model_dict["feature_type"] = __read_var(lines[idx], str)
    idx += 1
    model_dict["model_type"] = __read_var(lines[idx])
    idx += 1
    model_dict["max_p"] = __read_var(lines[idx])
    idx += 1
    model_dict["max_l"] = __read_var(lines[idx])
    idx += 1

    if model_dict["feature_type"] == "gtinv":
        gtinv_dict["order"] = __read_var(lines[idx])
        idx += 1
        gtinv_dict["max_l"] = __read_var(lines[idx], return_list=True)
        idx += 1
        gtinv_dict["sym"] = __read_var(lines[idx], strtobool, return_list=True)
        idx += 1
    else:
        gtinv_dict["order"] = 0
        gtinv_dict["max_l"] = []
        gtinv_dict["sym"] = []
        gtinv_dict["lm_seq"] = []
        gtinv_dict["l_comb"] = []
        gtinv_dict["lm_coeffs"] = []
        model_dict["max_l"] = 0

    # n_coeffs = __read_var(lines[idx])
    _ = __read_var(lines[idx])
    idx += 1
    mlp_dict["coeffs"] = np.array(__read_var(lines[idx], float, return_list=True))
    idx += 1
    mlp_dict["scales"] = np.array(__read_var(lines[idx], float, return_list=True))
    idx += 1

    n_pair_params = __read_var(lines[idx])
    idx += 1
    model_dict["pair_params"] = []
    for n in range(n_pair_params):
        params = __read_var(lines[idx], float, return_list=True)
        model_dict["pair_params"].append(params)
        idx += 1

    params_dict["mass"] = __read_var(lines[idx], float, return_list=True)
    idx += 1

    params_dict["electrostatic"] = __read_var(lines[idx], strtobool)
    idx += 1

    if model_dict["feature_type"] == "gtinv":
        try:
            if "gtinv_version" in lines[idx]:
                gtinv_dict["version"] = __read_var(lines[idx], int)
                idx += 1
            else:
                gtinv_dict["version"] = 1
        except:
            gtinv_dict["version"] = 1

        rgi = libmlpcpp.Readgtinv(
            gtinv_dict["order"],
            gtinv_dict["max_l"],
            gtinv_dict["sym"],
            params_dict["n_type"],
            gtinv_dict["version"],
        )
        gtinv_dict["lm_seq"] = rgi.get_lm_seq()
        gtinv_dict["l_comb"] = rgi.get_l_comb()
        gtinv_dict["lm_coeffs"] = rgi.get_lm_coeffs()

    return params_dict, mlp_dict


def load_mlp_lammps_flexible(file_list_or_file):

    if isinstance(file_list_or_file, list):
        if len(file_list_or_file) > 1:
            params_dicts, mlp_dicts = [], []
            for pot in file_list_or_file:
                params, mlp = load_mlp_lammps(pot)
                params_dicts.append(params)
                mlp_dicts.append(mlp)
            return params_dicts, mlp_dicts
        else:
            return load_mlp_lammps(file_list_or_file[0])
    else:
        return load_mlp_lammps(file_list_or_file)

    return params_dicts, mlp_dicts
