#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np

from pypolymlp.core.io_polymlp import save_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.accuracy import write_error_yaml
from pypolymlp.mlp_gen.features_attr import write_polymlp_params_yaml
from pypolymlp.mlp_gen.multi_datasets.accuracy import compute_error
from pypolymlp.mlp_gen.multi_datasets.parser import parse_observations
from pypolymlp.mlp_gen.multi_datasets.sequential import Sequential
from pypolymlp.mlp_gen.regression import Regression


def run_sequential_generator_multiple_datasets_from_params(
    params_dict, path_output="./"
):

    train_dft_dict, test_dft_dict = parse_observations(params_dict)

    t1 = time.time()
    seq_train = Sequential(params_dict, train_dft_dict)
    train_reg_dict = seq_train.get_updated_regression_dict()
    seq_test = Sequential(params_dict, test_dft_dict, scales=train_reg_dict["scales"])
    test_reg_dict = seq_test.get_updated_regression_dict()

    t2 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    coeffs, scales = reg.ridge_seq()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(
        params_dict,
        coeffs,
        scales,
        filename=path_output + "/polymlp.lammps",
    )

    print("  regression: best model")
    print("    alpha: ", mlp_dict["alpha"])

    t3 = time.time()
    error_dict = dict()
    error_dict["train"], error_dict["test"] = dict(), dict()

    for set_id, dft_dict in train_dft_dict.items():
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["train"][set_id] = compute_error(
            params_dict,
            coeffs,
            scales,
            dft_dict,
            output_key=output_key,
            path_output=path_output,
        )
    for set_id, dft_dict in test_dft_dict.items():
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["test"][set_id] = compute_error(
            params_dict,
            coeffs,
            scales,
            dft_dict,
            output_key=output_key,
            path_output=path_output,
        )

    t4 = time.time()

    mlp_dict["error"] = error_dict
    write_error_yaml(error_dict["train"], filename=path_output + "/polymlp_error.yaml")
    write_error_yaml(
        error_dict["test"],
        initialize=False,
        filename=path_output + "/polymlp_error.yaml",
    )
    write_polymlp_params_yaml(
        params_dict, filename=path_output + "/polymlp_params.yaml"
    )

    print("  elapsed_time:")
    print("    features + weighting: ", "{:.3f}".format(t2 - t1), "(s)")
    print("    regression:           ", "{:.3f}".format(t3 - t2), "(s)")
    print("    predictions:          ", "{:.3f}".format(t4 - t3), "(s)")

    return mlp_dict


def run_sequential_generator_multiple_datasets(infile):

    p = ParamsParser(infile, multiple_datasets=True)
    params_dict = p.get_params()

    mlp_dict = run_sequential_generator_multiple_datasets_from_params(params_dict)
    return mlp_dict


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        default="polymlp.in",
        help="Input file name",
    )
    args = parser.parse_args()

    run_sequential_generator_multiple_datasets(args.infile)
