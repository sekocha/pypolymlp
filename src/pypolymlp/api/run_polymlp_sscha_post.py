"""Command lines for post SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_post import PolymlpSSCHAPost


def run():
    """Command lines for post SSCHA calculations."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--properties",
        action="store_true",
        help="Calculate thermodynamic properties.",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Calculate properties of structures sampled from FC2 density matrix.",
    )
    parser.add_argument(
        "--transition",
        nargs=2,
        type=str,
        help="Find phase transition.",
    )
    parser.add_argument(
        "--boundary",
        nargs=2,
        type=str,
        help="Compute phase boundary.",
    )
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
    )
    parser.add_argument(
        "--fc2",
        type=str,
        default="fc2.hdf5",
        help="fc2.hdf5 file to be parsed.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp.lammps file",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=100,
        help="Number of sample supercells",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    sscha = PolymlpSSCHAPost(verbose=True)
    if args.properties:
        sscha.compute_thermodynamic_properties(
            args.yaml, filename="sscha_properties.yaml"
        )

    elif args.distribution:
        sscha.init_structure_distribution(
            yamlfile=args.yaml[0],
            fc2file=args.fc2,
            pot=args.pot,
        )
        sscha.run_structure_distribution(n_samples=args.n_samples)
        sscha.save_structure_distribution(path=".")
    elif args.transition:
        tc_linear, tc_quartic = sscha.find_phase_transition(
            args.transition[0],
            args.transition[1],
        )
        print("Tc (Linear fit)  :", np.round(tc_linear, 1))
        print("Tc (Quartic fit) :", np.round(tc_quartic, 1))

    elif args.boundary:
        sscha.compute_phase_boundary(
            args.boundary[0],
            args.boundary[1],
        )
