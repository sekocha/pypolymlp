"""Functions for calculating learning curves."""

import copy

import numpy as np

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY
from pypolymlp.mlp_dev.standard.regression import Regression


def find_slices(train_xy, total_n_atoms, n_samples):
    """"""

    ids = list(range(n_samples))

    first_id = train_xy.first_indices[0][2]
    ids_stress = range(first_id, first_id + n_samples * 6)
    ids.extend(ids_stress)

    first_id = train_xy.first_indices[0][1]
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

    print("Learning Curve:", flush=True)
    for n_samples, error in error_all:
        print("- n_samples:   ", n_samples, flush=True)
        print("  rmse_energy: ", "{:.8f}".format(error["energy"] * 1000), flush=True)
        print("  rmse_force:  ", "{:.8f}".format(error["force"]), flush=True)
        print("  rmse_stress: ", error["stress"], flush=True)


def learning_curve(
    polymlp: PolymlpDevDataXY,
    total_n_atoms: np.ndarray,
    verbose: bool = False,
):

    if len(polymlp.train) > 1:
        raise ValueError("A single dataset is required " "for learning curve option")

    train_xy = polymlp.train_xy
    polymlp_copy = copy.deepcopy(polymlp)

    error_all = []
    n_train = train_xy.first_indices[0][2]
    if verbose:
        print("Calculating learning curve...", flush=True)
    for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
        if verbose:
            print("------------- n_samples:", n_samples, "-------------", flush=True)
        ids = find_slices(train_xy, total_n_atoms, n_samples)

        polymlp_copy.train_xy.x = train_xy.x[ids]
        polymlp_copy.train_xy.y = train_xy.y[ids]
        polymlp_copy.train_xy.weight = train_xy.weight[ids]
        polymlp_copy.train_xy.scales = train_xy.scales

        reg = Regression(polymlp_copy).fit()
        acc = PolymlpDevAccuracy(reg)
        acc.compute_error()
        for error in acc.error_test_dict.values():
            error_all.append((n_samples, error))

    if verbose:
        write_learning_curve(error_all)
    return error_all
