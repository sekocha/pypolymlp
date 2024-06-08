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
        help="Use normal feature calculations",
    )
    parser.add_argument(
        "--learning_curve",
        action="store_true",
        help="Learning curve calculations",
    )
    args = parser.parse_args()

    verbose = True

    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(args.infile, verbose=True)
    polymlp_in.parse_datasets()
    polymlp_in.write_polymlp_params_yaml(filename="polymlp_params.yaml")

    if args.learning_curve:
        if len(polymlp_in.train_dict) == 1:
            args.no_sequential = True
            polymlp = PolymlpDevDataXY(polymlp_in).run()
            learning_curve(polymlp)
        else:
            raise ValueError(
                "A single dataset is required " "for learning curve option"
            )

    t1 = time.time()
    if args.no_sequential:
        if not args.learning_curve:
            polymlp = PolymlpDevDataXY(polymlp_in).run()
        polymlp.print_data_shape()
    else:
        polymlp = PolymlpDevDataXYSequential(polymlp_in).run()
    t2 = time.time()

    reg = Regression(polymlp).fit(seq=not args.no_sequential)
    reg.save_mlp_lammps(filename="polymlp.lammps")
    t3 = time.time()

    if verbose:
        mlp_dict = reg.best_model
        print("  Regression: best model")
        print("    alpha: ", mlp_dict["alpha"])

    acc = PolymlpDevAccuracy(reg)
    acc.compute_error()
    acc.write_error_yaml(filename="polymlp_error.yaml")
    t4 = time.time()

    if verbose:
        print("  elapsed_time:")
        print("    features:          ", "{:.3f}".format(t2 - t1), "(s)")
        print("    regression:        ", "{:.3f}".format(t3 - t2), "(s)")
        print("    error:             ", "{:.3f}".format(t4 - t3), "(s)")
