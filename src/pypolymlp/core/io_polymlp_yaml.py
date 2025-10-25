"""Class for saving and loading polymlp.yaml files"""

import io
from typing import Union

import numpy as np
import yaml

from pypolymlp.core.data_format import (
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParams,
)
from pypolymlp.core.utils import mass_table


def save_mlp_yaml(
    params: PolymlpParams,
    coeffs: np.ndarray,
    scales: np.ndarray,
    filename: bool = "polymlp.yaml",
):
    """Generate polymlp.yaml file for single polymlp model"""
    model = params.model

    np.set_printoptions(legacy="1.21")
    f = open(filename, "w")
    elements_str = "[" + ", ".join(["{0}".format(x) for x in params.elements]) + "]"
    print("elements:     ", elements_str, file=f)
    print("cutoff:       ", model.cutoff, file=f)
    print("pair_type:    ", model.pair_type, file=f)
    print("feature_type: ", model.feature_type, file=f)
    print("model_type:   ", model.model_type, file=f)
    print("max_p:        ", model.max_p, file=f)
    print("max_l:        ", model.max_l, file=f)
    print("", file=f)

    if model.feature_type == "gtinv":
        gtinv = model.gtinv
        print("gtinv_order:  ", gtinv.order, file=f)
        print("gtinv_maxl:   ", list(gtinv.max_l), file=f)
        print("gtinv_sym:    ", [0 for _ in gtinv.max_l], file=f)
        print("gtinv_version:", gtinv.version, file=f)
        print("", file=f)

    print("electrostatic:", 0, file=f)
    mass = [mass_table()[ele] for ele in params.elements]
    print("mass:         ", mass, file=f)
    print("", file=f)

    print("n_pair_params:", len(model.pair_params), file=f)
    print("pair_params:", file=f)
    for obj in model.pair_params:
        print("-", list(obj), file=f)
    print("", file=f)

    print("n_type_pairs:", len(model.pair_params_conditional), file=f)
    print("type_pairs:", file=f)
    for atomtypes, n_ids in model.pair_params_conditional.items():
        print("- atom_type_pair:     ", list(atomtypes), file=f)
        print("  pair_params_indices:", list(n_ids), file=f)
    print("", file=f)

    if params.type_full is not None:
        print("type_full:   ", int(params.type_full), file=f)
        print("type_indices:", list(params.type_indices), file=f)
    else:
        print("type_full:   ", 1, file=f)
        print("type_indices:", list(np.arange(params.n_type)), file=f)
    print("", file=f)

    coeffs_ = coeffs / scales
    coeffs_str = "[" + ", ".join([f"{c:.15e}" for c in coeffs_]) + "]"
    print("n_coeffs:", len(coeffs), file=f)
    print("coeffs:", coeffs_str, file=f)

    f.close()


def load_mlp_yaml(filename: Union[str, io.IOBase] = "polymlp.yaml"):
    """Load polymlp.yaml file.

    Return
    ------
    params: Parameters in PolymlpParams.
    coeffs: polymlp model coefficients.

    Usage
    -----
    params, coeffs = load_mlp_yaml(filename='polymlp.yaml')
    """

    if isinstance(filename, io.IOBase):
        yml = yaml.safe_load(filename)
    else:
        yml = yaml.safe_load(open(filename))

    elements = yml["elements"]
    element_order = elements
    n_type = len(elements)
    mass = yml["mass"]

    if yml["feature_type"] == "gtinv":
        gtinv = PolymlpGtinvParams(
            order=yml["gtinv_order"],
            max_l=yml["gtinv_maxl"],
            sym=yml["gtinv_sym"],
            n_type=n_type,
            version=yml["gtinv_version"],
        )
    else:
        gtinv = PolymlpGtinvParams(order=0, max_l=[], sym=[], n_type=n_type)

    pair_params_cond = dict()
    for tp in yml["type_pairs"]:
        pair_params_cond[tuple(tp["atom_type_pair"])] = tp["pair_params_indices"]

    model = PolymlpModelParams(
        cutoff=yml["cutoff"],
        model_type=yml["model_type"],
        max_p=yml["max_p"],
        max_l=yml["max_l"],
        feature_type=yml["feature_type"],
        gtinv=gtinv,
        pair_type=yml["pair_type"],
        pair_conditional=True,
        pair_params=yml["pair_params"],
        pair_params_conditional=pair_params_cond,
    )
    params = PolymlpParams(
        n_type=n_type,
        elements=elements,
        model=model,
        element_order=element_order,
        type_full=yml["type_full"],
        type_indices=yml["type_indices"],
        mass=mass,
    )
    return params, yml["coeffs"]
