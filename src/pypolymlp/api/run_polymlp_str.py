"""Command lines for generating DFT structures."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.utils import print_credit


def run():
    """Command lines for generating DFT structures."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscars",
        type=str,
        nargs="*",
        required=True,
        help="Initial structures in POSCAR format",
    )
    parser.add_argument(
        "--max_natom",
        type=int,
        default=150,
        help="Maximum number of atoms in structures",
    )

    parser.add_argument(
        "--displacements",
        type=int,
        default=None,
        help="Number of structures sampled using displacements.",
    )
    parser.add_argument(
        "--isotropic",
        type=int,
        default=None,
        help="Number of structures sampled using isotropic volume changes.",
    )
    parser.add_argument(
        "--dense_equilibrium",
        action="store_true",
        help="Use dense grid around equilibrium volume.",
    )

    parser.add_argument(
        "--standard",
        type=int,
        default=None,
        help=("Number of structures sampled using a standard algorithm."),
    )
    parser.add_argument(
        "--low_density",
        type=int,
        default=None,
        help="Number of structures sampled using low density mode.",
    )
    parser.add_argument(
        "--high_density",
        type=int,
        default=None,
        help="Number of structures sampled using high density mode.",
    )

    parser.add_argument(
        "--distance_density_mode",
        type=float,
        default=0.2,
        help="Maximum distance of distributions for generating atomic "
        "displacements in low and high density modes",
    )
    parser.add_argument(
        "--const_distance",
        type=float,
        default=None,
        help="Constant magnitude of distributions for generating displacements",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=None,
        help="Maximum magnitude of distributions "
        "for generating displacements and cell distortions",
    )

    # This option is activated only when --displacements option is used.
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=(1, 1, 1),
        help=("Supercell size for random displacement" " generation"),
    )
    # The following options are activated
    # only when --displacements and --max_distance options are used.
    parser.add_argument("--n_volumes", type=int, default=1, help="Number of volumes.")
    parser.add_argument(
        "--min_volume", type=float, default=0.8, help="Minimum volume ratio."
    )
    parser.add_argument(
        "--max_volume", type=float, default=1.3, help="Maximumm volume ratio."
    )

    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    print_credit()
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_structures_from_files(poscars=args.poscars)

    if args.displacements is not None:
        print("Pypolymlp structure generator: Displacement mode", flush=True)
        if args.const_distance is None and args.max_distance is None:
            raise RuntimeError("const_distance or max_distance is required.")

        polymlp.build_supercell(supercell_size=args.supercell)
        if args.const_distance is not None:
            polymlp.run_const_displacements(
                n_samples=args.displacements,
                distance=args.const_distance,
            )
        elif args.max_distance is not None:
            polymlp.run_sequential_displacements(
                n_samples=args.displacements,
                distance_lb=0.01,
                distance_ub=args.max_distance,
                n_volumes=args.n_volumes,
                eps_min=args.min_volume,
                eps_max=args.max_volume,
            )
    elif args.isotropic is not None:
        print("Pypolymlp structure generator: Isotropic volume changes", flush=True)
        polymlp.build_supercell(supercell_size=args.supercell)
        polymlp.run_isotropic_volume_changes(
            n_samples=args.isotropic,
            eps_min=args.min_volume,
            eps_max=args.max_volume,
            dense_equilibrium=args.dense_equilibrium,
        )
    else:
        print("Pypolymlp structure generator: Standard algorithms", flush=True)
        polymlp.build_supercells_auto(max_natom=129)
        if args.standard is not None:
            print("Run standard algorithms", flush=True)
            if args.const_distance is not None:
                print(
                    "const_distance is activated only for --displacements option",
                    flush=True,
                )
            if args.max_distance is None:
                raise RuntimeError("max_distance is required.")

            polymlp.run_standard_algorithm(
                n_samples=args.standard,
                max_distance=args.max_distance,
            )
        if args.low_density is not None:
            print("Run low-density algorithm", flush=True)
            polymlp.run_density_algorithm(
                n_samples=args.low_density,
                distance=args.distance_density_mode,
                vol_algorithm="low_auto",
            )
        if args.high_density is not None:
            print("Run high-density algorithm", flush=True)
            polymlp.run_density_algorithm(
                n_samples=args.high_density,
                distance=args.distance_density_mode,
                vol_algorithm="high_auto",
            )
    polymlp.save_random_structures(path="./poscars")
    print(polymlp.n_samples, "structures are generated.", flush=True)
