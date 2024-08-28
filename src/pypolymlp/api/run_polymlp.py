#!/usr/bin/env python
import argparse
import signal
import time

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.standard.learning_curve import learning_curve
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression


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

    verbose = True
    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(args.infile, verbose=verbose)
    polymlp_in.parse_datasets()
    polymlp_in.write_polymlp_params_yaml(filename="polymlp_params.yaml")
    n_features = polymlp_in.n_features

    if args.learning_curve:
        t1 = time.time()
        if len(polymlp_in.train_dict) == 1:
            args.no_sequential = True
            polymlp = PolymlpDevDataXY(polymlp_in).run()
            learning_curve(polymlp)
        else:
            raise ValueError(
                "A single dataset is required " "for learning curve option"
            )
        polymlp.print_data_shape()
        t2 = time.time()
    else:
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
            if verbose:
                polymlp.print_data_shape()
        else:
            polymlp = PolymlpDevDataXYSequential(polymlp_in, verbose=verbose).run_train(
                batch_size=batch_size
            )
        t2 = time.time()

    reg = Regression(polymlp).fit(
        seq=not args.no_sequential,
        clear_data=True,
        batch_size=batch_size,
    )
    reg.save_mlp_lammps(filename="polymlp.lammps")
    t3 = time.time()

    if verbose:
        mlp = reg.best_model
        print("  Regression: best model", flush=True)
        print("    alpha: ", mlp.alpha, flush=True)

    acc = PolymlpDevAccuracy(reg)
    acc.compute_error()
    acc.write_error_yaml(filename="polymlp_error.yaml")
    t4 = time.time()

    if verbose:
        print("  elapsed_time:", flush=True)
        print("    features:          ", "{:.3f}".format(t2 - t1), "(s)", flush=True)
        print("    regression:        ", "{:.3f}".format(t3 - t2), "(s)", flush=True)
        print("    error:             ", "{:.3f}".format(t4 - t3), "(s)", flush=True)
