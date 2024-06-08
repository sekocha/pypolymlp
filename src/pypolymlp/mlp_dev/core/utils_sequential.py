#!/usr/bin/env python


def get_batch_slice(n_data, batch_size):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def slice_dft_dict(dft_dict, begin, end):
    dft_dict_sliced = dict()
    dft_dict_sliced["structures"] = dft_dict["structures"][begin:end]
    dft_dict_sliced["energy"] = dft_dict["energy"][begin:end]

    begin_f = sum(dft_dict["total_n_atoms"][:begin]) * 3
    end_f = sum(dft_dict["total_n_atoms"][:end]) * 3
    dft_dict_sliced["force"] = dft_dict["force"][begin_f:end_f]

    dft_dict_sliced["stress"] = dft_dict["stress"][begin * 6 : end * 6]
    dft_dict_sliced["volumes"] = dft_dict["volumes"]
    dft_dict_sliced["elements"] = dft_dict["elements"]
    dft_dict_sliced["total_n_atoms"] = dft_dict["total_n_atoms"][begin:end]
    dft_dict_sliced["include_force"] = dft_dict["include_force"]
    dft_dict_sliced["weight"] = dft_dict["weight"]

    return dft_dict_sliced
