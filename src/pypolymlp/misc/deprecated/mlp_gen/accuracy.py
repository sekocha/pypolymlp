#!/usr/bin/env python
import os

import numpy as np

from pypolymlp.core.utils import rmse


def __compute_rmse(
    true_values,
    pred_values_all,
    weight_all,
    begin_id,
    end_id,
    normalize=None,
    return_values=False,
):

    pred_values = pred_values_all[begin_id:end_id]
    weight = weight_all[begin_id:end_id]

    pred = pred_values / weight

    if normalize is None:
        true = true_values
    else:
        true = true_values / np.array(normalize)
        pred /= np.array(normalize)
    if return_values:
        return rmse(true, pred), true, pred
    return rmse(true, pred)


def print_error(error_dict, key="train"):

    print("  prediction:", key)
    print(
        "    rmse_energy:",
        "{0:13.4f}".format(error_dict["energy"] * 1000),
        "(meV/atom)",
    )
    if error_dict["force"] is not None:
        print(
            "    rmse_force: ",
            "{0:13.4f}".format(error_dict["force"]),
            "(eV/ang)",
        )
    if error_dict["stress"] is not None:
        print(
            "    rmse_stress:",
            "{0:13.4f}".format(error_dict["stress"] * 1000),
            "(meV/atom)",
        )


def compute_error(
    dft_dict,
    params_dict,
    predictions_all,
    weights_all,
    first_indices,
    output_key="train",
    log_force=False,
    path_output="./",
):

    if "include_force" in dft_dict:
        include_force = dft_dict["include_force"]
    else:
        include_force = params_dict["include_force"]

    if include_force == False:
        include_stress = False
    else:
        include_stress = params_dict["include_stress"]

    n_data = len(predictions_all)
    ebegin, fbegin, sbegin = first_indices
    eend = ebegin + len(dft_dict["energy"])
    if include_force:
        fend = fbegin + len(dft_dict["force"])
        send = sbegin + len(dft_dict["stress"])

    n_total_atoms = [sum(st["n_atoms"]) for st in dft_dict["structures"]]
    rmse_e, true_e, pred_e = __compute_rmse(
        dft_dict["energy"],
        predictions_all,
        weights_all,
        ebegin,
        eend,
        normalize=n_total_atoms,
        return_values=True,
    )

    rmse_f, rmse_s = None, None
    if include_force:
        rmse_f = __compute_rmse(
            dft_dict["force"], predictions_all, weights_all, fbegin, fend
        )

    if include_stress:
        stress_unit = "eV"
        if stress_unit == "eV":
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == "GPa":
            eV_to_GPa = 160.21766208
            volumes = [st["volume"] for st in dft_dict["structures"]]
            normalize = np.repeat(volumes, 6) / eV_to_GPa
        rmse_s = __compute_rmse(
            dft_dict["stress"],
            predictions_all,
            weights_all,
            sbegin,
            send,
            normalize=normalize,
        )

    error_dict = dict()
    error_dict["energy"] = rmse_e
    error_dict["force"] = rmse_f
    error_dict["stress"] = rmse_s
    print_error(error_dict, key=output_key)

    filenames = dft_dict["filenames"]
    outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T

    os.makedirs(path_output + "/predictions", exist_ok=True)
    f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
    print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
    for d, name in zip(outdata, filenames):
        print(d[0], d[1], d[2], name, file=f)
    f.close()

    if log_force:
        _, true_f, pred_f = __compute_rmse(
            dft_dict["force"],
            predictions_all,
            weights_all,
            fbegin,
            fend,
            return_values=True,
        )
        outdata = np.array([true_f, pred_f, (true_f - pred_f)]).T

        f = open(path_output + "/predictions/force." + output_key + ".dat", "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()

    return error_dict


def write_error_yaml(error_dict, filename="polymlp_error.yaml", initialize=True):

    if initialize:
        f = open(filename, "w")
        print("units:", file=f)
        print("  energy: meV/atom", file=f)
        print("  force:  eV/angstrom", file=f)
        print("  stress: meV/atom", file=f)
        print("", file=f)
        print("prediction_errors:", file=f)
        f.close()

    f = open(filename, "a")
    for key, dict1 in error_dict.items():
        print("- dataset:", key, file=f)
        print("  rmse_energy: ", dict1["energy"] * 1000, file=f)
        if dict1["force"] is not None:
            print("  rmse_force:  ", dict1["force"], file=f)
        if dict1["stress"] is not None:
            print("  rmse_stress: ", dict1["stress"] * 1000, file=f)
        print("", file=f)
    f.close()
