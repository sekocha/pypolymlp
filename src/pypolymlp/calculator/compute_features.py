"""Functions for computing structural features."""

import argparse
import signal
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_dev.core.features import Features


def update_types(structures: list[PolymlpStructure], element_order: list[str]):
    """Reorder types in PolymlpStructure.

    Integers in types will be compatible with element_order.
    """
    for st in structures:
        types = np.ones(len(st.types), dtype=int) * 1000
        elements = np.array(st.elements)
        for i, ele in enumerate(element_order):
            types[elements == ele] = i
        st.types = types
        if np.any(types == 1000):
            print("elements (structure) =", st.elements)
            print("elements (polymlp.lammps) =", element_order)
            raise ("Elements in structure are not found in polymlp.lammps")
    return structures


def compute_from_polymlp_lammps(
    structures: list[PolymlpStructure],
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    force: bool = False,
    stress: bool = False,
    return_mlp_dict: bool = True,
    return_features_obj: bool = False,
):
    """Compute features from polymlp.lammps file or PolymlpParams object.

    Parameters
    ----------
    structures: Structures.
    pot: polymlp file.
    params: Parameters for polymlp.

    Any one of pot and params is required.
    """
    if pot is not None:
        if len(pot) > 1:
            raise NotImplementedError("Only single polymlp.lammps file is available.")
        params, mlp_dict = load_mlp_lammps(filename=pot[0])

    params.include_force = force
    params.include_stress = stress

    element_order = params.elements
    structures = update_types(structures, element_order)
    features = Features(params, structures=structures, print_memory=False)

    if return_features_obj and return_mlp_dict:
        return features, mlp_dict
    elif return_features_obj and not return_mlp_dict:
        return features
    elif not return_features_obj and return_mlp_dict:
        return features.x, mlp_dict
    return features.x


def compute_from_infile(
    infile: str,
    structures: list[PolymlpStructure],
    force: bool = None,
    stress: bool = None,
):
    """Compute features from polymlp.in.

    Parameters
    ----------
    infile: Input parameter file.
    structures: Structures.
    force: Generate force structural features.
    stress: Generate stress structural features.

    example:
    > $(pypolymlp)/calculator/compute_features.py
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
    params = ParamsParser(infile, parse_vasprun_locations=False).params
    if force is not None:
        params.include_force = force
    if stress is not None:
        params.include_stress = stress
    element_order = params.elements

    structures = update_types(structures, element_order)
    features = Features(params, structures=structures, print_memory=False)
    return features.x


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
