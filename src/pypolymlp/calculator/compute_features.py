#!/usr/bin/env python
import argparse
import signal

import numpy as np

from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.features import Features


def update_types(st_dicts, element_order):

    for st in st_dicts:
        types = np.ones(len(st["types"]), dtype=int) * 1000
        elements = np.array(st["elements"])
        for i, ele in enumerate(element_order):
            types[elements == ele] = i
        st["types"] = types
        if np.any(types == 1000):
            print("elements (structure) =", st["elements"])
            print("elements (polymlp.lammps) =", element_order)
            raise ("Elements in structure are not found in polymlp.lammps")

    return st_dicts


def compute_from_polymlp_lammps(
    st_dicts,
    pot=None,
    params_dict=None,
    force=False,
    stress=False,
    return_mlp_dict=True,
    return_features_obj=False,
):

    if pot is not None:
        params_dict, mlp_dict = load_mlp_lammps(filename=pot)

    params_dict["include_force"] = force
    params_dict["include_stress"] = stress

    element_order = params_dict["elements"]
    st_dicts = update_types(st_dicts, element_order)
    features = Features(params_dict, st_dicts, print_memory=False)

    if return_features_obj and return_mlp_dict:
        return features, mlp_dict
    elif return_features_obj and not return_mlp_dict:
        return features
    elif not return_features_obj and return_mlp_dict:
        return features.get_x(), mlp_dict
    return features.get_x()


def compute_from_infile(infile, st_dicts, force=None, stress=None):
    """
    > example: $(pypolymlp)/calculator/compute_features.py
                    --infile polymlp.in --poscars poscars/poscar-000*
    > cat polymlp.in

        n_type 2
        elements Mg O
        feature_type gtinv
        cutoff 8.0
        model_type 3
        max_p 2
        gtinv_order 3
        gtinv_maxl 4 4
        gaussian_params1 1.0 1.0 1
        gaussian_params2 0.0 7.0 8
    """
    p = ParamsParser(infile, parse_vasprun_locations=False)
    params_dict = p.get_params()
    if force is not None:
        params_dict["include_force"] = force
    if stress is not None:
        params_dict["include_stress"] = stress
    element_order = params_dict["elements"]

    st_dicts = update_types(st_dicts, element_order)
    features = Features(params_dict, st_dicts, print_memory=False)
    return features.get_x()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscars",
        nargs="*",
        type=str,
        default=["POSCAR"],
        help="poscar files",
    )
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        default=None,
        help="Input parameter settings",
    )
    parser.add_argument(
        "--pot",
        type=str,
        default=None,
        help="Input parameter settings (polymlp.lammps)",
    )
    args = parser.parse_args()

    structures = parse_structures_from_poscars(args.poscars)
    if args.infile is not None:
        x = compute_from_infile(args.infile, structures)
    elif args.pot is not None:
        x = compute_from_polymlp_lammps(structures, pot=args.pot, return_mlp_dict=False)

    print(" feature size =", x.shape)
    np.save("features.npy", x)
