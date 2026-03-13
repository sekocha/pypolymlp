"""Command lines for transfer learning."""

import argparse
import signal
import time

import numpy as np

from pypolymlp.core.io_polymlp import load_mlps
from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.transfer.regression_transfer import RegressionTransfer

if __name__ == "__main__":

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
        help="Use normal feature calculations.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=["polymlp.lammps"],
        help="MLP file used for regularization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size of feature calculations",
    )
    args = parser.parse_args()

    verbose = True
    """TODO: params and polymlp_in.params must be the same.
    If they are different, exception must be returned.
    """
    params, coeffs = load_mlps(args.pot)

    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(args.infile, verbose=verbose)
    polymlp_in.parse_datasets()
    polymlp_in.write_polymlp_params_yaml(filename="polymlp_params.yaml")
    n_features = polymlp_in.n_features

    batch_size = None
    if not args.no_sequential:
        if args.batch_size is None:
            batch_size = max((10000000 // n_features), 128)
        else:
            batch_size = args.batch_size
        if verbose:
            print("Batch size:", batch_size, flush=True)

    t1 = time.time()
    if args.no_sequential:
        polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
        polymlp.print_data_shape()
    else:
        polymlp = PolymlpDevDataXYSequential(polymlp_in, verbose=verbose).run_train(
            batch_size=batch_size
        )
    t2 = time.time()

    reg = RegressionTransfer(polymlp)
    reg.fit(
        coeffs,
        np.ones(len(coeffs)),
        seq=not args.no_sequential,
        clear_data=True,
        batch_size=batch_size,
    )
    reg.save_mlp(filename="polymlp.yaml")
    t3 = time.time()

    if verbose:
        mlp = reg.best_model
        print("  Regression: best model")
        print("    alpha: ", mlp.alpha)
        print("    beta: ", mlp.beta)

    acc = PolymlpDevAccuracy(reg)
    acc.compute_error()
    acc.write_error_yaml(filename="polymlp_error.yaml")
    t4 = time.time()

    if verbose:
        print("  elapsed_time:")
        print("    features:          ", "{:.3f}".format(t2 - t1), "(s)")
        print("    regression:        ", "{:.3f}".format(t3 - t2), "(s)")
        print("    error:             ", "{:.3f}".format(t4 - t3), "(s)")
