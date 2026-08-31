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
        "--batch_size",
        type=int,
        default=100,
        help="Batch size of online regression",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-2,
        help="Tolerance for gradient in online regression",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
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
    polymlp.load_mlp(args.pot)
    if verbose:
        polymlp.print_params()

    # TODO: Reconsider dataset interface
    polymlp.set_datasets_vasp(vaspruns=args.vaspruns)

    t1 = time.time()
    polymlp.fit_online(
        batch_size=args.batch_size,
        gtol=args.gtol,
        n_epochs=args.n_epochs,
    )

    polymlp.save_mlp(filename="polymlp.yaml.update")
    t2 = time.time()
    # polymlp.estimate_error(log_energy=True, use_cv=args.cross_val)
    # t3 = time.time()
    # polymlp.save_errors(filename="polymlp_error.yaml")

    if verbose:
        print("Regression: update model", flush=True)
        print("elapsed_time:", flush=True)
        print("  features, fit:      ", "{:.3f}".format(t2 - t1), "(s)", flush=True)
        # print("  error:              ", "{:.3f}".format(t3 - t2), "(s)", flush=True)
