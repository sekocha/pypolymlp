"""Command lines for generating FC basis set using symfc."""

import argparse
import signal
import time

import numpy as np
from symfc import Symfc

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.symfc_utils import set_symfc_cutoffs, structure_to_symfc_cell


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poscar",
        type=str,
        default=None,
        help="Structure in POSCAR format.",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
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

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    unitcell = Poscar(args.poscar).structure
    supercell = supercell_diagonal(
        unitcell,
        args.supercell,
        use_phonopy=args.use_phonopy,
    )
    supercell = structure_to_symfc_cell(supercell)
    cutoff = set_symfc_cutoffs(args.cutoff_fc2, args.cutoff_fc3, args.cutoff_fc4)

    t1 = time.time()
    symfc = Symfc(supercell, use_mkl=not args.disable_mkl, log_level=1, cutoff=cutoff)
    symfc.compute_basis_set(orders=args.orders)
    t2 = time.time()

    print("FC orders:", tuple(args.orders), flush=True)
    print("Elapsed time (Basis sets):", "{:.3f}".format(t2 - t1), flush=True)
    for order in args.orders:
        print(
            "Number of FC basis vectors (order " + str(order) + "):",
            symfc.basis_set[order].blocked_basis_set.shape[1],
            flush=True,
        )
