"""Command lines for generating FC basis set using symfc."""

import argparse
import signal
import time

import numpy as np
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc import Symfc

from pypolymlp.core.interface_vasp import Poscar, Vasprun, parse_forces_displacements
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.symfc_utils import set_symfc_cutoffs, structure_to_symfc_cell


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poscar",
        type=str,
        required=True,
        help="Structure in POSCAR format.",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        required=True,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--orders",
        nargs="*",
        type=int,
        default=(2, 3),
        help="Orders of force constants.",
    )
    parser.add_argument(
        "--cutoff_fc2",
        type=float,
        default=None,
        help="Cutoff radius for FC2.",
    )
    parser.add_argument(
        "--cutoff_fc3",
        type=float,
        default=None,
        help="Cutoff radius for FC3.",
    )
    parser.add_argument(
        "--cutoff_fc4",
        type=float,
        default=None,
        help="Cutoff radius for FC4.",
    )
    parser.add_argument(
        "--disable_mkl",
        action="store_true",
        help="Disable to use MKL.",
    )
    parser.add_argument(
        "--use_phonopy",
        action="store_true",
        help="Use phonopy to make supercell.",
    )

    parser.add_argument(
        "--vaspruns",
        nargs="*",
        type=str,
        default=None,
        help="vasprun.xml files of FC training data.",
    )
    parser.add_argument(
        "--vasprun_residual",
        type=str,
        default=None,
        help="vasprun.xml file used for residual forces",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for FC solver.",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    if args.vaspruns is not None:
        args.use_phonopy = True

    unitcell = Poscar(args.poscar).structure
    supercell = supercell_diagonal(
        unitcell,
        args.supercell,
        use_phonopy=args.use_phonopy,
    )
    supercell_symfc = structure_to_symfc_cell(supercell)
    cutoff = set_symfc_cutoffs(args.cutoff_fc2, args.cutoff_fc3, args.cutoff_fc4)

    t1 = time.time()
    use_mkl = not args.disable_mkl
    symfc = Symfc(supercell_symfc, use_mkl=use_mkl, log_level=1, cutoff=cutoff)
    symfc.compute_basis_set(orders=args.orders)
    t2 = time.time()

    print("FC orders:", tuple(args.orders), flush=True)
    print("Elapsed time (Basis sets):", "{:.3f}".format(t2 - t1), flush=True)
    for order in args.orders:
        prefix = "Number of FC basis vectors (order " + str(order) + "):"
        print(prefix, symfc.basis_set[order].blocked_basis_set.shape[1], flush=True)

    if args.vaspruns is not None:
        forces, disps = parse_forces_displacements(args.vaspruns, supercell)
        if args.vasprun_residual is not None:
            vasp = Vasprun(args.vasprun_residual)
            for f in forces:
                f -= vasp.forces

        symfc.forces = forces.transpose((0, 2, 1))
        symfc.displacements = disps.transpose((0, 2, 1))
        symfc.solve(orders=args.orders, batch_size=args.batch_size, is_compact_fc=True)
        if symfc.force_constants[2] is not None:
            print("Writing fc2.hdf5", flush=True)
            write_fc2_to_hdf5(symfc.force_constants[2])

        if symfc.force_constants[3] is not None:
            print("Writing fc3.hdf5", flush=True)
            write_fc3_to_hdf5(symfc.force_constants[3])
