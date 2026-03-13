"""Class for saving and loading polymlp.lammps files"""

import io
import itertools

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParamsSingle,
)
from pypolymlp.core.utils import strtobool


def _read_var(line: str, dtype=int, return_list: bool = False):
    list1 = line.split("#")[0].split()
    if return_list:
        return [dtype(v) for v in list1]
    return dtype(list1[0])


def load_mlp_lammps(filename: str = "polymlp.lammps"):
    """Load polymlp.lammps file.

    Return
    ------
    params: Parameters in PolymlpParamsSingle.
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

    mass = _read_var(lines[idx], float, return_list=True)  # mass
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
    params = PolymlpParamsSingle(
        n_type=n_type,
        elements=elements,
        model=model,
        element_order=element_order,
        type_full=type_full,
        type_indices=type_indices,
        mass=mass,
    )
    coeffs_ = coeffs / scales
    return params, coeffs_
