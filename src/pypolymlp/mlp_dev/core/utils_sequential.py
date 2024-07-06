#!/usr/bin/env python

import numpy as np


def get_batch_slice(n_data, batch_size):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def slice_dft_dict(dft_dict, begin, end):
    begin_f = sum(dft_dict["total_n_atoms"][:begin]) * 3
    end_f = sum(dft_dict["total_n_atoms"][:end]) * 3
    dft_dict_sliced = {
        "structures": dft_dict["structures"][begin:end],
        "energy": dft_dict["energy"][begin:end],
        "force": dft_dict["force"][begin_f:end_f],
        "stress": dft_dict["stress"][begin * 6 : end * 6],
        "volumes": dft_dict["volumes"][begin:end],
        "total_n_atoms": dft_dict["total_n_atoms"][begin:end],
        "elements": dft_dict["elements"],
        "include_force": dft_dict["include_force"],
        "weight": dft_dict["weight"],
    }
    return dft_dict_sliced


def sort_dft_dict(dft_dict):
    ids = np.argsort(dft_dict["total_n_atoms"])
    ids_stress = ((ids * 6)[:, None] + np.arange(6)[None, :]).reshape(-1)
    force_end = np.cumsum(dft_dict["total_n_atoms"] * 3)
    force_begin = np.insert(force_end[:-1], 0, 0)
    ids_force = np.array(
        [i for b, e in zip(force_begin[ids], force_end[ids]) for i in range(b, e)]
    )
    dft_dict_sorted = {
        "structures": [dft_dict["structures"][i] for i in ids],
        "energy": dft_dict["energy"][ids],
        "force": dft_dict["force"][ids_force],
        "stress": dft_dict["stress"][ids_stress],
        "volumes": dft_dict["volumes"][ids],
        "total_n_atoms": dft_dict["total_n_atoms"][ids],
        "elements": dft_dict["elements"],
        "include_force": dft_dict["include_force"],
        "weight": dft_dict["weight"],
    }
    return dft_dict_sorted
