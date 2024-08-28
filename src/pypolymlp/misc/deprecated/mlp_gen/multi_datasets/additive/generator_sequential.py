#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np

from pypolymlp.core.io_polymlp import save_multiple_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.accuracy import write_error_yaml
from pypolymlp.mlp_gen.features_attr import write_polymlp_params_yaml
from pypolymlp.mlp_gen.multi_datasets.accuracy import compute_error
from pypolymlp.mlp_gen.multi_datasets.additive.params_utils import (
    print_common_params,
    set_common_params_dict,
)
from pypolymlp.mlp_gen.multi_datasets.additive.sequential import Sequential
from pypolymlp.mlp_gen.multi_datasets.parser import parse_observations
from pypolymlp.mlp_gen.regression import Regression


def run_sequential_generator_additive(infiles):

    multiple_params_dicts = [
        ParamsParser(infile, multiple_datasets=True).get_params() for infile in infiles
    ]
    common_params_dict = set_common_params_dict(multiple_params_dicts)
    print_common_params(common_params_dict, infile=infiles[0])

    train_dft_dict, test_dft_dict = parse_observations(common_params_dict)

    t1 = time.time()
    seq_train = Sequential(multiple_params_dicts, train_dft_dict)
    train_reg_dict = seq_train.get_updated_regression_dict()
    seq_test = Sequential(
        multiple_params_dicts,
        test_dft_dict,
        scales=train_reg_dict["scales"],
    )
    test_reg_dict = seq_test.get_updated_regression_dict()

    t2 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, common_params_dict)
    coeffs, scales = reg.ridge_seq()
    mlp_dict = reg.get_best_model()

    multiple_coeffs, multiple_scales = save_multiple_mlp_lammps(
        multiple_params_dicts,
        train_reg_dict["cumulative_n_features"],
        coeffs,
        scales,
    )

    print("  regression: best model")
    print("    alpha: ", mlp_dict["alpha"])

    t3 = time.time()
    error_dict = dict()
    error_dict["train"], error_dict["test"] = dict(), dict()

    for set_id, dft_dict in train_dft_dict.items():
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["train"][set_id] = compute_error(
            multiple_params_dicts,
            multiple_coeffs,
            multiple_scales,
            dft_dict,
            output_key=output_key,
        )
    for set_id, dft_dict in test_dft_dict.items():
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["test"][set_id] = compute_error(
            multiple_params_dicts,
            multiple_coeffs,
            multiple_scales,
            dft_dict,
            output_key=output_key,
        )

    t4 = time.time()
    mlp_dict["error"] = error_dict
    write_error_yaml(error_dict["train"])
    write_error_yaml(error_dict["test"], initialize=False)

    for i, params_dict in enumerate(multiple_params_dicts):
        filename = "polymlp_params" + str(i + 1) + ".yaml"
        write_polymlp_params_yaml(params_dict, filename=filename)

    print("  elapsed_time:")
    print("    features + weighting: ", "{:.3f}".format(t2 - t1), "(s)")
    print("    regression:           ", "{:.3f}".format(t3 - t2), "(s)")
    print("    predictions:          ", "{:.3f}".format(t4 - t3), "(s)")

    return mlp_dict


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
    args = parser.parse_args()

    run_sequential_generator_additive(args.infile)
