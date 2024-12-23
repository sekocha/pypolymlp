"""Command lines for generating structures used for systematic SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_str import PolymlpSSCHAStructureGenerator


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
    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    strgen = PolymlpSSCHAStructureGenerator(verbose=True)
    strgen.load_poscar(args.poscar)
    strgen.run(
        n_samples=args.n_samples,
        max_deform=args.max_deform,
        max_distance=args.max_distance,
    )
    strgen.save_random_structures(path="./poscars")
    print(args.n_samples, "structures are generated.", flush=True)
