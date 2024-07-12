#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np

from pypolymlp.core.interface_vasp import parse_vaspruns
from pypolymlp.core.io_polymlp import save_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.accuracy import compute_error, write_error_yaml
from pypolymlp.mlp_gen.features import Features
from pypolymlp.mlp_gen.features_attr import write_polymlp_params_yaml
from pypolymlp.mlp_gen.learning_curve import learning_curve
from pypolymlp.mlp_gen.precondition import Precondition
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
    - train (vasprun locations)
    - test (vasprun locations)

Variables in dft_dict (train_dft_dict, test_dft_dict):
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
    - elements
    - volumes
    - total_n_atoms

Variables in reg_dict
    - x
    - y
    - first_indices [(ebegin, fbegin, sbegin), ...]
    - n_data (ne, nf, ns)
    - scales
"""


def run_generator_single_dataset_from_params_and_datasets(
    params_dict,
    train_dft_dict,
    test_dft_dict,
    log=True,
    compute_learning_curve=False,
    path_output="./",
):

    t1 = time.time()
    features_train = Features(params_dict, train_dft_dict["structures"])
    train_reg_dict = features_train.get_regression_dict()
    features_test = Features(params_dict, test_dft_dict["structures"])
    test_reg_dict = features_test.get_regression_dict()

    t2 = time.time()
    pre_train = Precondition(train_reg_dict, train_dft_dict, params_dict, scales=None)
    pre_train.print_data_shape(header="training data size")
    train_reg_dict = pre_train.get_updated_regression_dict()

    pre_test = Precondition(
        test_reg_dict, test_dft_dict, params_dict, scales=train_reg_dict["scales"]
    )
    pre_test.print_data_shape(header="test data size")
    test_reg_dict = pre_test.get_updated_regression_dict()

    t3 = time.time()

    if compute_learning_curve:
        total_n_atoms = train_dft_dict["total_n_atoms"]
        error_all = learning_curve(
            train_reg_dict, test_reg_dict, test_dft_dict, params_dict, total_n_atoms
        )
        f = open("polymlp_learning_curve.dat", "w")
        print(
            "# n_str, RMSE(energy, meV/atom), " "RMSE(force, eV/ang), RMSE(stress)",
            file=f,
        )
        for n_samp, error in error_all:
            print(
                n_samp, error["energy"] * 1000, error["force"], error["stress"], file=f
            )
        f.close()

        if log:
            print("Learning Curve:")
            for n_samples, error in error_all:
                print("- n_samples:   ", n_samples)
                print("  rmse_energy: ", "{:.8f}".format(error["energy"] * 1000))
                print("  rmse_force:  ", "{:.8f}".format(error["force"]))
                print("  rmse_stress: ", error["stress"])

    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    # coeffs, scales = reg.ridge()
    coeffs, scales = reg.test_random_projection()
    # coeffs, scales = reg.test_pca_projection()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(
        params_dict, coeffs, scales, filename=path_output + "/polymlp.lammps"
    )

    if log:
        print("  regression: best model")
        print("    alpha: ", mlp_dict["alpha"])

    t4 = time.time()
    error_dict = dict()
    indices = train_reg_dict["first_indices"][0]
    error_dict["train"] = compute_error(
        train_dft_dict,
        params_dict,
        mlp_dict["predictions"]["train"],
        train_reg_dict["weight"],
        indices,
        output_key="train",
        path_output=path_output,
    )
    indices = test_reg_dict["first_indices"][0]
    error_dict["test"] = compute_error(
        test_dft_dict,
        params_dict,
        mlp_dict["predictions"]["test"],
        test_reg_dict["weight"],
        indices,
        output_key="test",
        path_output=path_output,
    )
    write_error_yaml(error_dict, filename=path_output + "/polymlp_error.yaml")
    write_polymlp_params_yaml(
        params_dict, filename=path_output + "/polymlp_params.yaml"
    )
    mlp_dict["error"] = error_dict

    if log:
        print("  elapsed_time:")
        print("    features:          ", "{:.3f}".format(t2 - t1), "(s)")
        print("    scaling, weighting:", "{:.3f}".format(t3 - t2), "(s)")
        print("    regression:        ", "{:.3f}".format(t4 - t3), "(s)")

    return mlp_dict


def run_generator_single_dataset_from_params(
    params_dict,
    log=True,
    compute_learning_curve=False,
    path_output="./",
):

    if params_dict["dataset_type"] == "vasp":
        train_dft_dict = parse_vaspruns(
            params_dict["dft"]["train"], element_order=params_dict["element_order"]
        )
        test_dft_dict = parse_vaspruns(
            params_dict["dft"]["test"], element_order=params_dict["element_order"]
        )
    elif params_dict["dataset_type"] == "phono3py":
        from pypolymlp.core.interface_phono3py import parse_phono3py_yaml

        train_dft_dict = parse_phono3py_yaml(
            params_dict["dft"]["train"]["phono3py_yaml"],
            params_dict["dft"]["train"]["energy"],
            element_order=params_dict["element_order"],
            select_ids=params_dict["dft"]["train"]["indices"],
            use_phonon_dataset=False,
        )
        test_dft_dict = parse_phono3py_yaml(
            params_dict["dft"]["test"]["phono3py_yaml"],
            params_dict["dft"]["test"]["energy"],
            element_order=params_dict["element_order"],
            select_ids=params_dict["dft"]["test"]["indices"],
            use_phonon_dataset=False,
        )

    mlp_dict = run_generator_single_dataset_from_params_and_datasets(
        params_dict,
        train_dft_dict,
        test_dft_dict,
        log=log,
        compute_learning_curve=compute_learning_curve,
        path_output=path_output,
    )
    return mlp_dict


def run_generator_single_dataset(infile, log=True, compute_learning_curve=False):

    p = ParamsParser(infile)
    params_dict = p.get_params()
    mlp_dict = run_generator_single_dataset_from_params(
        params_dict,
        log=log,
        compute_learning_curve=compute_learning_curve,
    )

    return mlp_dict


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--infile", type=str, default="polymlp.in", help="Input file name"
    )
    args = parser.parse_args()

    run_generator_single_dataset(args.infile)
