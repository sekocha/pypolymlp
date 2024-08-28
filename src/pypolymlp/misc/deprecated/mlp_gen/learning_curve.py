#!/usr/bin/env python
import numpy as np

from pypolymlp.mlp_gen.accuracy import compute_error
from pypolymlp.mlp_gen.regression import Regression


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


def learning_curve(
    train_reg_dict, test_reg_dict, test_dft_dict, params_dict, total_n_atoms
):

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
        reg = Regression(train_reg_dict_sample, test_reg_dict, params_dict)
        _, _ = reg.ridge()
        mlp_dict = reg.get_best_model()

        indices = test_reg_dict["first_indices"][0]
        error = compute_error(
            test_dft_dict,
            params_dict,
            mlp_dict["predictions"]["test"],
            test_reg_dict["weight"],
            indices,
            output_key="test",
        )
        error_all.append((n_samples, error))

    return error_all
