"""Command lines for developing polynomial MLP from file."""

import argparse
import signal
import time

import numpy as np

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        nargs="*",
        type=str,
        default=["polymlp.in"],
        help="Input file name",
    )
    parser.add_argument(
        "--no_sequential",
        action="store_true",
        help="Use normal feature calculations",
    )
    parser.add_argument(
        "--learning_curve",
        action="store_true",
        help="Learning curve calculations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size of feature calculations",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    verbose = True
    polymlp = Pypolymlp()
    polymlp.load_parameter_file(args.infile)
    polymlp.parse_datasets()
    polymlp.save_parameters(filename="polymlp_params.yaml")

    if args.learning_curve:
        tlearn1 = time.time()
        polymlp.fit_learning_curve(verbose=verbose)
        polymlp.save_learning_curve(filename="polymlp_learning_curve.dat")
        tlearn2 = time.time()

    t1 = time.time()
    polymlp.fit(
        sequential=not args.no_sequential,
        batch_size=args.batch_size,
        verbose=verbose,
    )
    polymlp.save_mlp(filename="polymlp.lammps")
    t2 = time.time()
    polymlp.estimate_error(log_energy=True, verbose=verbose)
    t3 = time.time()
    polymlp.save_errors(filename="polymlp_error.yaml")

    if verbose:
        print("Regression: best model", flush=True)
        print("  alpha: ", polymlp.summary.alpha, flush=True)
        print("elapsed_time:", flush=True)
        if args.learning_curve:
            print(
                "  learning curve:     ",
                "{:.3f}".format(tlearn2 - tlearn1),
                "(s)",
                flush=True,
            )
        print("  features, fit:      ", "{:.3f}".format(t2 - t1), "(s)", flush=True)
        print("  error:              ", "{:.3f}".format(t3 - t2), "(s)", flush=True)
