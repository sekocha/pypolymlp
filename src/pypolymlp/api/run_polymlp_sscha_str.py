"""Command lines for generating structures used for systematic SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_str import PypolymlpSSCHAStructureGenerator
from pypolymlp.core.utils import print_credit


def run():
    """Command lines for generating structures for systematic SSCHA calculations."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        required=True,
        help="Initial structure in POSCAR format",
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Generate structures with symmetric constraints.",
    )
    parser.add_argument(
        "--volume",
        action="store_true",
        help="Generate structures with volumes changed.",
    )
    parser.add_argument(
        "--cell",
        action="store_true",
        help="Generate structures with cell shapes changed.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of sample structures.",
    )
    parser.add_argument(
        "--max_deform",
        type=float,
        default=0.1,
        help="Maximum magnitude of lattice deformation.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=0.1,
        help="Maximum magnitude of atomic displacements.",
    )
    parser.add_argument(
        "--min_volume",
        type=float,
        default=0.8,
        help="Minimum volume ratio.",
    )
    parser.add_argument(
        "--max_volume",
        type=float,
        default=1.3,
        help="Maximumm volume ratio.",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    if not args.sym and not args.volume and not args.cell:
        args.sym = True

    strgen = PypolymlpSSCHAStructureGenerator(verbose=True)
    strgen.load_poscar(args.poscar)

    if args.sym:
        strgen.sample_sym(
            n_samples=args.n_samples,
            max_deform=args.max_deform,
            max_distance=args.max_distance,
        )
    elif args.volume:
        strgen.sample_volumes(
            n_samples=args.n_samples,
            eps_min=args.min_volume,
            eps_max=args.max_volume,
            fix_axis=True,
        )
    elif args.cell:
        strgen.sample_volumes(
            n_samples=args.n_samples,
            eps_min=args.min_volume,
            eps_max=args.max_volume,
            fix_axis=False,
            max_deform=args.max_deform,
        )

    strgen.save_structures(path="./poscars")

    print(args.n_samples, "structures are generated.", flush=True)
    if args.sym:
        basis_axis, basis_cartesian = strgen.basis_sets
        if basis_axis is not None:
            print("axis:", basis_axis.shape[0], "DOFs", flush=True)
            for i, b in enumerate(basis_axis):
                print("Basis", i + 1, ":", flush=True)
                print(b, flush=True)
        if basis_cartesian is not None:
            print("positions:", basis_cartesian.shape[0], "DOFs", flush=True)
            for i, b in enumerate(basis_cartesian):
                print("Basis", i + 1, ":", flush=True)
                print(b.T, flush=True)
