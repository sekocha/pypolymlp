#!/usr/bin/env python
import itertools
import os

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_gen.accuracy import print_error


def __compute_rmse(true_values, pred_values, normalize=None, return_values=False):

    if normalize is None:
        true = true_values
        pred = pred_values
    else:
        true = true_values / np.array(normalize)
        pred = pred_values / np.array(normalize)

    if return_values:
        return rmse(true, pred), true, pred
    return rmse(true, pred)


def compute_error(
    params_dict,
    coeffs,
    scales,
    dft_dict,
    output_key="train",
    stress_unit="eV",
    log_force=False,
    log_stress=False,
    path_output="./",
):

    if isinstance(params_dict, list) and len(params_dict) > 1:
        coeffs_rescale = [c / s for c, s in zip(coeffs, scales)]
    else:
        coeffs_rescale = coeffs / scales

    prop = Properties(params_dict=params_dict, coeffs=coeffs_rescale)
    energies, forces, stresses = prop.eval_multiple(dft_dict["structures"])
    forces = np.array(
        list(itertools.chain.from_iterable([f.T.reshape(-1) for f in forces]))
    )
    stresses = stresses.reshape(-1)

    n_total_atoms = [sum(st["n_atoms"]) for st in dft_dict["structures"]]
    rmse_e, true_e, pred_e = __compute_rmse(
        dft_dict["energy"],
        energies,
        normalize=n_total_atoms,
        return_values=True,
    )

    if log_force == False:
        rmse_f = __compute_rmse(dft_dict["force"], forces)
    else:
        rmse_f, true_f, pred_f = __compute_rmse(
            dft_dict["force"], forces, return_values=True
        )

    if stress_unit == "eV":
        normalize = np.repeat(n_total_atoms, 6)
    elif stress_unit == "GPa":
        eV_to_GPa = 160.21766208
        volumes = [st["volume"] for st in dft_dict["structures"]]
        normalize = np.repeat(volumes, 6) / eV_to_GPa

    if log_stress == False:
        rmse_s = __compute_rmse(dft_dict["stress"], stresses, normalize=normalize)
    else:
        rmse_s, true_s, pred_s = __compute_rmse(
            dft_dict["stress"],
            stresses,
            normalize=normalize,
            return_values=True,
        )

    error_dict = dict()
    error_dict["energy"] = rmse_e
    error_dict["force"] = rmse_f
    error_dict["stress"] = rmse_s
    print_error(error_dict, key=output_key)

    os.makedirs(path_output + "/predictions", exist_ok=True)
    filenames = dft_dict["filenames"]

    outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
    f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
    print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
    for d, name in zip(outdata, filenames):
        print(d[0], d[1], d[2], name, file=f)
    f.close()

    if log_force:
        outdata = np.array([true_f, pred_f, (true_f - pred_f)]).T
        f = open(path_output + "/predictions/force." + output_key + ".dat", "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()

    if log_stress:
        outdata = np.array([true_s, pred_s, (true_s - pred_s)]).T
        f = open(path_output + "/predictions/stress." + output_key + ".dat", "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()

    return error_dict
