#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np

from pypolymlp.core.io_polymlp import save_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.accuracy import compute_error, write_error_yaml
from pypolymlp.mlp_gen.features_attr import write_polymlp_params_yaml
from pypolymlp.mlp_gen.multi_datasets.features import Features
from pypolymlp.mlp_gen.multi_datasets.parser import parse_observations
from pypolymlp.mlp_gen.multi_datasets.precondition import Precondition
from pypolymlp.mlp_gen.regression import Regression

"""
    Variables in params_dict:
      - n_type
      - include_force
      - include_stress
      - model
        - cutoff
        - model_type
        - max_p
        - max_l
        - feature_type
        - pair_type
        - pair_params
        - gtinv
          - order
          - max_l
          - lm_seq
          - l_comb
          - lm_coeffs
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train
            - dataset1
                - vaspruns
                - include_force
                - weight
                - atomtype
            - dataset2
        - test
            - ...

    Variables in dft_dict (train_dft_dict, test_dft_dict):
        multiple_dft_dicts
        - dataset1
          dft_dict:
            - energy
            - force
            - stress
            - structures
              - structure (1)
                - axis
                - positions
                - n_atoms
                - types
                - elements
              - ...
            - vaspruns
            - include_force
            - weight
            - atomtype
        - dataset2 ...

    Variables in reg_dict
      - train
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
      - test
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
      - scaler

"""


def run_generator_multiple_datasets_from_params(params_dict, path_output="./"):

    train_dft_dict, test_dft_dict = parse_observations(params_dict)

    t1 = time.time()
    features_train = Features(params_dict, train_dft_dict)
    train_reg_dict = features_train.get_regression_dict()

    features_test = Features(params_dict, test_dft_dict)
    test_reg_dict = features_test.get_regression_dict()

    t2 = time.time()
    pre_train = Precondition(train_reg_dict, train_dft_dict, params_dict, scales=None)
    pre_train.print_data_shape(header="training data size")
    train_reg_dict = pre_train.get_updated_regression_dict()

    pre_test = Precondition(
        test_reg_dict,
        test_dft_dict,
        params_dict,
        scales=train_reg_dict["scales"],
    )
    pre_test.print_data_shape(header="test data size")
    test_reg_dict = pre_test.get_updated_regression_dict()

    t3 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    coeffs, scales = reg.ridge()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(
        params_dict,
        coeffs,
        scales,
        filename=path_output + "/polymlp.lammps",
    )

    print("  regression: best model")
    print("    alpha: ", mlp_dict["alpha"])

    t4 = time.time()
    error_dict = dict()
    error_dict["train"], error_dict["test"] = dict(), dict()
    for (set_id, dft_dict), indices in zip(
        train_dft_dict.items(), train_reg_dict["first_indices"]
    ):
        predictions = mlp_dict["predictions"]["train"]
        weights = train_reg_dict["weight"]
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["train"][set_id] = compute_error(
            dft_dict,
            params_dict,
            predictions,
            weights,
            indices,
            output_key=output_key,
            path_output=path_output,
        )

    for (set_id, dft_dict), indices in zip(
        test_dft_dict.items(), test_reg_dict["first_indices"]
    ):
        predictions = mlp_dict["predictions"]["test"]
        weights = test_reg_dict["weight"]
        output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace("..", "")
        error_dict["test"][set_id] = compute_error(
            dft_dict,
            params_dict,
            predictions,
            weights,
            indices,
            output_key=output_key,
            path_output=path_output,
        )

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
    print("    features:          ", "{:.3f}".format(t2 - t1), "(s)")
    print("    scaling, weighting:", "{:.3f}".format(t3 - t2), "(s)")
    print("    regression:        ", "{:.3f}".format(t4 - t3), "(s)")

    return mlp_dict


def run_generator_multiple_datasets(infile):

    p = ParamsParser(infile, multiple_datasets=True)
    params_dict = p.get_params()

    mlp_dict = run_generator_multiple_datasets_from_params(params_dict)
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

    run_generator_multiple_datasets(args.infile)
