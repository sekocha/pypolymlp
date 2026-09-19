"""Command lines for developing polynomial MLP from file using online algorithm."""

import argparse
import signal
import time

import numpy as np

from pypolymlp.core.utils import print_credit
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    # TODO: Hybrid
    parser.add_argument(
        "--pot",
        type=str,
        default="polymlp.yaml",
        help="Polymlp file name.",
    )
    parser.add_argument(
        "--max_learning_rate",
        type=float,
        default=1e-3,
        help="Maximum learning rate used as the initial one.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Magnitude parameter for regularization.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.99,
        help="Parameter for defining gradient update.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Mini-batch size of online Adam",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-5,
        help="Tolerance for gradient in online regression",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000,
        help="Number of epochs",
    )
    parser.add_argument(
        "--vaspruns",
        nargs="*",
        type=str,
        required=True,
        help="vasprun.xml files used for updating MLP",
    )

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    verbose = True
    polymlp = Pypolymlp(verbose=verbose)
    polymlp.load_mlp(args.pot, require_atomic_energy=True)
    if verbose:
        polymlp.print_params()

    polymlp.set_datasets_vasp_online(vaspruns=args.vaspruns)

    t1 = time.time()
    polymlp.fit_online(
        max_learning_rate=args.max_learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        gtol=args.gtol,
        n_epochs=args.n_epochs,
    )
    polymlp.save_mlp(filename="polymlp.yaml.update")
    t2 = time.time()

    if verbose:
        print("Regression: update model", flush=True)
        print("elapsed_time:", flush=True)
        print("  features, fit:      ", "{:.3f}".format(t2 - t1), "(s)", flush=True)
