"""Class for saving and loading polymlp.lammps files"""

import io
import itertools

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParams,
)
from pypolymlp.core.utils import mass_table, strtobool


def _read_var(line, dtype=int, return_list=False):
    list1 = line.split("#")[0].split()
    if return_list:
        return [dtype(v) for v in list1]
    return dtype(list1[0])


def load_mlp_lammps(filename="polymlp.lammps"):
    """Load polymlp.lammps file.

    Return
    ------
    params: Parameters in PolymlpParams.
    coeffs: polymlp model coefficients.

    Usage
    -----
    params, coeffs = load_mlp_lammps(filename='polymlp.lammps')
    """

    if isinstance(filename, io.IOBase):
        lines = filename.readlines()
    else:
        with open(filename) as f:
            lines = f.readlines()

    idx = 0
    elements = _read_var(lines[idx], str, return_list=True)
    element_order = elements
    n_type = len(elements)
    idx += 1

    cutoff = _read_var(lines[idx], float)
    idx += 1
    pair_type = _read_var(lines[idx], str)
    idx += 1
    feature_type = _read_var(lines[idx], str)
    idx += 1
    model_type = _read_var(lines[idx])
    idx += 1
    max_p = _read_var(lines[idx])
    idx += 1
    max_l = _read_var(lines[idx])
    idx += 1

    if feature_type == "gtinv":
        gtinv_order = _read_var(lines[idx])
        idx += 1
        gtinv_maxl = _read_var(lines[idx], return_list=True)
        idx += 1
        gtinv_sym = _read_var(lines[idx], strtobool, return_list=True)
        idx += 1
    else:
        gtinv_order = 0
        gtinv_maxl = []
        gtinv_sym = []
        max_l = 0

    _ = _read_var(lines[idx])
    idx += 1
    coeffs = np.array(_read_var(lines[idx], float, return_list=True))
    idx += 1
    scales = np.array(_read_var(lines[idx], float, return_list=True))
    idx += 1

    n_pair_params = _read_var(lines[idx])
    idx += 1
    pair_params = []
    for n in range(n_pair_params):
        params = _read_var(lines[idx], float, return_list=True)
        pair_params.append(params)
        idx += 1

    _ = _read_var(lines[idx], float, return_list=True)  # mass
    idx += 1

    _ = _read_var(lines[idx], strtobool)  # electrostatic
    idx += 1

    if feature_type == "gtinv":
        try:
            if "gtinv_version" in lines[idx]:
                gtinv_version = _read_var(lines[idx], int)
                idx += 1
            else:
                gtinv_version = 1
        except:
            gtinv_version = 1
    else:
        gtinv_version = 1

    pair_params_cond = dict()
    try:
        if "n_type_pairs" in lines[idx]:
            pair_cond = True
            n_type_pairs = _read_var(lines[idx], int)
            idx += 1
            for _ in range(n_type_pairs):
                atomtypes = _read_var(lines[idx], int, return_list=True)[0:2]
                idx += 1
                params = _read_var(lines[idx], int, return_list=True)
                idx += 1
                pair_params_cond[tuple(atomtypes)] = params
    except:
        pair_cond = False
        for atomtypes in itertools.combinations_with_replacement(range(n_type), 2):
            pair_params_cond[atomtypes] = list(range(n_pair_params))

    try:
        if "type_full" in lines[idx]:
            type_full = _read_var(lines[idx], strtobool)
            idx += 1
            type_indices = _read_var(lines[idx], int, return_list=True)
            idx += 1
    except:
        type_full = True
        type_indices = list(range(n_type))

    gtinv = PolymlpGtinvParams(
        order=gtinv_order,
        max_l=gtinv_maxl,
        sym=gtinv_sym,
        n_type=n_type,
        version=gtinv_version,
    )
    model = PolymlpModelParams(
        cutoff=cutoff,
        model_type=model_type,
        max_p=max_p,
        max_l=max_l,
        feature_type=feature_type,
        gtinv=gtinv,
        pair_type=pair_type,
        pair_conditional=pair_cond,
        pair_params=pair_params,
        pair_params_conditional=pair_params_cond,
    )
    params = PolymlpParams(
        n_type=n_type,
        elements=elements,
        model=model,
        element_order=element_order,
        type_full=type_full,
        type_indices=type_indices,
    )
    coeffs_ = coeffs / scales
    return params, coeffs_


def _print_param(dict1, key, fstream, prefix=""):
    print(str(dict1[key]), "#", prefix + key, file=fstream)


def _print_array1d(array, fstream, comment="", fmt=None):
    for obj in array:
        if fmt is not None:
            print(fmt.format(obj), end=" ", file=fstream)
        else:
            print(obj, end=" ", file=fstream)
    print("#", comment, file=fstream)


def save_mlp_lammps(
    params: PolymlpParams,
    coeffs: np.ndarray,
    scales: np.ndarray,
    filename: bool = "polymlp.lammps",
):
    """Generate polymlp.lammps file for single polymlp model"""
    f = open(filename, "w")
    _print_array1d(params.elements, f, comment="elements")
    model_dict = params.model.as_dict()
    _print_param(model_dict, "cutoff", f)
    _print_param(model_dict, "pair_type", f)
    _print_param(model_dict, "feature_type", f)
    _print_param(model_dict, "model_type", f)
    _print_param(model_dict, "max_p", f)
    _print_param(model_dict, "max_l", f)

    if model_dict["feature_type"] == "gtinv":
        gtinv_dict = model_dict["gtinv"]
        _print_param(gtinv_dict, "order", f, prefix="gtinv_")
        _print_array1d(gtinv_dict["max_l"], f, comment="gtinv_max_l")
        gtinv_sym = [0 for _ in gtinv_dict["max_l"]]
        _print_array1d(gtinv_sym, f, comment="gtinv_sym")

    print(len(coeffs), "# n_coeffs", file=f)
    _print_array1d(coeffs, f, comment="reg. coeffs", fmt="{0:15.15e}")
    _print_array1d(scales, f, comment="scales", fmt="{0:15.15e}")

    print(len(model_dict["pair_params"]), "# n_params", file=f)
    for obj in model_dict["pair_params"]:
        print(
            "{0:15.15f}".format(obj[0]),
            "{0:15.15f}".format(obj[1]),
            "# pair func. params",
            file=f,
        )

    mass = [mass_table()[ele] for ele in params.elements]
    _print_array1d(mass, f, comment="atomic mass", fmt="{0:15.15e}")
    print("False # electrostatic", file=f)

    if model_dict["feature_type"] == "gtinv":
        _print_param(gtinv_dict, "version", f, prefix="gtinv_")

    print(len(model_dict["pair_params_conditional"]), "# n_type_pairs", file=f)
    for atomtypes, n_ids in model_dict["pair_params_conditional"].items():
        for v in atomtypes:
            print(v, end=" ", file=f)
        print(len(n_ids), "# atom type pair", file=f)
        for v in n_ids:
            print(v, end=" ", file=f)
        print("# pair params indices ", file=f)

    if params.type_full is not None:
        print(int(params.type_full), "# type_full", file=f)
        _print_array1d(params.type_indices, f, comment="type_indices")
    else:
        print("1 # type_full", file=f)
        _print_array1d(np.arange(params.n_type), f, comment="type_indices")

    f.close()


def save_multiple_mlp_lammps(
    multiple_params: list[PolymlpParams],
    cumulative_n_features: int,
    coeffs: np.ndarray,
    scales: np.ndarray,
    prefix: str = "polymlp.lammps",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate polymlp.lammps files for hybrid polymlp model"""
    multiple_coeffs = []
    multiple_scales = []
    for i, params in enumerate(multiple_params):
        if i == 0:
            begin, end = 0, cumulative_n_features[0]
        else:
            begin, end = (
                cumulative_n_features[i - 1],
                cumulative_n_features[i],
            )

        save_mlp_lammps(
            params,
            coeffs[begin:end],
            scales[begin:end],
            filename=prefix + "." + str(i + 1),
        )
        multiple_coeffs.append(coeffs[begin:end])
        multiple_scales.append(scales[begin:end])

    return multiple_coeffs, multiple_scales
