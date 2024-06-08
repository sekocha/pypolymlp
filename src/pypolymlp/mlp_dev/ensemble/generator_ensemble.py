#!/usr/bin/env python
import argparse
import signal
import time

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.ensemble.mlpdev_dataxy_feature_bagging import (
    PolymlpDevDataXYFeatureBagging,
    PolymlpDevDataXYFeatureBaggingSequential,
)
from pypolymlp.mlp_dev.ensemble.regression_ensemble import RegressionEnsemble

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
        "-n",
        "--n_models",
        type=int,
        default=10,
        help="Number of ensemble models",
    )
    parser.add_argument(
        "-r",
        "--ratio_features",
        type=float,
        default=0.5,
        help="Ratio that gives the number of features in ensemble models.",
    )
    args = parser.parse_args()

    verbose = True
    polymlp_in = PolymlpDevData()
    polymlp_in.parse_infiles(args.infile, verbose=verbose)
    polymlp_in.parse_datasets()
    polymlp_in.write_polymlp_params_yaml(filename="polymlp_params.yaml")

    t1 = time.time()
    if args.no_sequential:
        polymlp = PolymlpDevDataXYFeatureBagging(polymlp_in)
    else:
        polymlp = PolymlpDevDataXYFeatureBaggingSequential(polymlp_in)

    polymlp.run(
        n_models=args.n_models,
        ratio_feature_samples=args.ratio_features,
    )
    t2 = time.time()

    reg = RegressionEnsemble(polymlp).fit()
    reg.save_mlp_lammps(filename="polymlp.lammps")

    t3 = time.time()

    acc = PolymlpDevAccuracy(reg)
    acc.compute_error()
    acc.write_error_yaml(filename="polymlp_error.yaml")
    t4 = time.time()

    if verbose:
        print("  elapsed_time:")
        print("    features:          ", "{:.3f}".format(t2 - t1), "(s)")
        print("    regression:        ", "{:.3f}".format(t3 - t2), "(s)")
        print("    error:             ", "{:.3f}".format(t4 - t3), "(s)")
