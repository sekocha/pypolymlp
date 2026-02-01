"""Compute FCs using a vasp dataset."""

import itertools

import numpy as np


def calc_displacements(positions_array, original_positions, axis):

    disps = []
    trans = np.array(list(itertools.product(*[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
    diff = positions_array - original_positions
    for d1 in diff:
        disp_frac = np.zeros(d1.shape)
        norms = np.ones(d1.shape[1]) * 1e10
        for t1 in trans:
            d2 = d1 - np.tile(t1, (d1.shape[1], 1)).T
            norms_d2 = np.linalg.norm(d2, axis=0)
            match = norms_d2 < norms
            norms[match] = norms_d2[match]
            disp_frac[:, match] = d2[:, match]
        disps.append(axis @ disp_frac)

    disps = np.array(disps)
    return disps


if __name__ == "__main__":

    """How to use.
    python3 ~/git/pypolymlp/src/pypolymlp/symfc/compute_fcs_from_vasp.py --str_yaml $dir/polymlp_str.yaml --vaspruns $dir/finished/{00001..00100}/vasprun.xml.polymlp --vasprun_residual $dir/finished/00000/vasprun.xml.polymlp
    """

    import argparse
    import signal

    from pypolymlp.calculator.fc import PolymlpFC
    from pypolymlp.core.interface_vasp import Poscar, Vasprun
    from pypolymlp.utils.phonopy_utils import (
        phonopy_cell_to_st_dict,
        phonopy_supercell,
        st_dict_to_phonopy_cell,
    )
    from pypolymlp.utils.yaml_utils import load_cells

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--str_yaml", type=str, default=None, help="polymlp_str.yaml file"
    )
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--vaspruns",
        nargs="*",
        type=str,
        default=None,
        required=True,
        help="vasprun.xml files",
    )
    parser.add_argument(
        "--vasprun_residual",
        type=str,
        default=None,
        help="vasprun.xml file for residual_forces",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for FC solver.",
    )

    args = parser.parse_args()

    if args.str_yaml is not None:
        _, supercell_dict = load_cells(filename=args.str_yaml)
        supercell = st_dict_to_phonopy_cell(supercell_dict)
    else:
        unitcell_dict = Poscar(args.poscar).get_structure()
        supercell_matrix = np.diag(args.supercell)
        supercell = phonopy_supercell(unitcell_dict, supercell_matrix)
        supercell_dict = phonopy_cell_to_st_dict(supercell)

    forces, positions = [], []
    for f in args.vaspruns:
        vasp = Vasprun(f)
        forces.append(vasp.forces)
        st = vasp.structure
        positions.append(st.positions)
    forces = np.array(forces)

    if args.vasprun_residual is not None:
        vasp = Vasprun(args.vasprun_residual)
        residual_forces = vasp.forces
        for f in forces:
            f -= residual_forces

    disps = calc_displacements(
        positions, supercell_dict["positions"], supercell_dict["axis"]
    )

    polyfc = PolymlpFC(supercell=supercell)
    polyfc.displacements = disps

    polyfc.run(forces=forces, batch_size=args.batch_size, sum_rule_basis=True)
