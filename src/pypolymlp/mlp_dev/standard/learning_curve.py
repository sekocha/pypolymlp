#!/usr/bin/env python
import copy

import numpy as np

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY
from pypolymlp.mlp_dev.standard.regression import Regression


def find_slices(train_reg_dict, total_n_atoms, n_samples):

    ids = list(range(n_samples))

    first_id = train_reg_dict["first_indices"][0][2]
    ids_stress = range(first_id, first_id + n_samples * 6)
    ids.extend(ids_stress)

    first_id = train_reg_dict["first_indices"][0][1]
    n_forces = sum(total_n_atoms[:n_samples]) * 3
    ids_force = range(first_id, first_id + n_forces)
    ids.extend(ids_force)

    return np.array(ids)


def write_learning_curve(error_all):

    f = open("polymlp_learning_curve.dat", "w")
    print(
        "# n_str, RMSE(energy, meV/atom), " "RMSE(force, eV/ang), RMSE(stress)",
        file=f,
    )
    for n_samp, error in error_all:
        print(
            n_samp,
            error["energy"] * 1000,
            error["force"],
            error["stress"],
            file=f,
        )
    f.close()

    print("Learning Curve:")
    for n_samples, error in error_all:
        print("- n_samples:   ", n_samples)
        print("  rmse_energy: ", "{:.8f}".format(error["energy"] * 1000))
        print("  rmse_force:  ", "{:.8f}".format(error["force"]))
        print("  rmse_stress: ", error["stress"])


def learning_curve(polymlp: PolymlpDevDataXY):

    if len(polymlp.train_dict) > 1:
        raise ValueError("A single dataset is required " "for learning curve option")

    train_reg_dict = polymlp.train_regression_dict

    for values in polymlp.train_dict.values():
        total_n_atoms = values["total_n_atoms"]

    polymlp_copy = copy.deepcopy(polymlp)

    error_all = []
    n_train = train_reg_dict["first_indices"][0][2]
    print("Calculating learning curve...")
    for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
        print("------------- n_samples:", n_samples, "-------------")
        ids = find_slices(train_reg_dict, total_n_atoms, n_samples)
        train_reg_dict_sample = {
            "x": train_reg_dict["x"][ids],
            "y": train_reg_dict["y"][ids],
            "weight": train_reg_dict["weight"][ids],
            "scales": train_reg_dict["scales"],
        }
        polymlp_copy.train_regression_dict = train_reg_dict_sample
        reg = Regression(polymlp_copy).fit()

        acc = PolymlpDevAccuracy(reg)
        acc.compute_error()
        for error in acc.error_test_dict.values():
            error_all.append((n_samples, error))

    write_learning_curve(error_all)
    return error_all
