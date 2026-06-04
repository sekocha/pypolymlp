"""Command lines for genumerating polynomial invariants."""

import argparse
import signal

import numpy as np

from pypolymlp.polyinv.api_polyinv import run_enum, save_coeffs, solve


def run():
    """Command lines for genumerating polynomial invariants."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orders",
        nargs="*",
        type=int,
        default=None,
        help="n-th order product.",
    )
    parser.add_argument(
        "--minl",
        type=int,
        default=None,
        help="Exclude l combinations composed of only l < minl.",
    )

    parser.add_argument("--maxl", type=int, default=10, help="Maximum l value.")
    parser.add_argument(
        "--l",
        nargs="*",
        type=int,
        default=None,
        help="Combination of l values.",
    )
    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")

    if args.orders:
        run_enum(args.orders, args.maxl, args.minl, lproj=0, verbose=True)
    elif args.l:
        eigvecs, lm_indices = solve(args.l, lproj=0, verbose=True)
        save_coeffs(eigvecs, lm_indices, filename="polyinv_coeffs.yaml", mode="w")
