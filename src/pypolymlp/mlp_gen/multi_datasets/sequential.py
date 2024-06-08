#!/usr/bin/env python
import gc

import numpy as np

from pypolymlp.mlp_gen.multi_datasets.features import Features
from pypolymlp.mlp_gen.precondition import apply_atomic_energy, apply_weight_percentage


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


class Sequential:

    def __init__(
        self,
        params_dict,
        multiple_dft_dicts,
        scales=None,
        verbose=True,
        element_swap=False,
        batch_size=64,
    ):

        self.multiple_dft_dicts = multiple_dft_dicts

        for _, dft_dict in self.multiple_dft_dicts.items():
            dft_dict = apply_atomic_energy(dft_dict, params_dict)
        min_e_per_atom = self.__find_min_energy()

        xtx, xty, y_sq_norm = None, None, 0.0
        xe_sum, xe_sq_sum = None, None
        total_n_data = 0
        for i, (set_id, dft_dict) in enumerate(multiple_dft_dicts.items()):
            if verbose:
                print("----- Dataset:", set_id, "-----")

            structures = dft_dict["structures"]

            begin_ids, end_ids = get_batch_slice(len(structures), batch_size)
            for begin, end in zip(begin_ids, end_ids):
                dft_dict_sliced = slice_dft_dict(dft_dict, begin, end)
                if verbose:
                    print("Number of structures:", end - begin)

                dft_dict_tmp = dict({"tmp": dft_dict_sliced})
                features = Features(
                    params_dict,
                    dft_dict_tmp,
                    print_memory=verbose,
                    element_swap=element_swap,
                )

                x = features.get_x()
                first_indices = features.get_first_indices()[0]

                if verbose:
                    ram = x.shape[1] * x.shape[1] * 8e-9 * 2
                    if i == 0:
                        print(
                            " Memory allocation (X^T @ X) :",
                            "{:.3f}".format(ram),
                            "(GB)",
                        )

                if scales is None:
                    xe = x[: features.ne]
                    local1 = np.sum(xe, axis=0)
                    local2 = np.sum(np.square(xe), axis=0)
                    xe_sum = self.__sum_array(xe_sum, local1)
                    xe_sq_sum = self.__sum_array(xe_sq_sum, local2)

                n_data, n_features = x.shape
                y = np.zeros(n_data)
                w = np.ones(n_data)
                total_n_data += n_data

                x, y, w = apply_weight_percentage(
                    x,
                    y,
                    w,
                    dft_dict_sliced,
                    params_dict,
                    first_indices,
                    min_e=min_e_per_atom,
                )
                xtx1 = x.T @ x
                xty1 = x.T @ y

                xtx = self.__sum_array(xtx, xtx1)
                xty = self.__sum_array(xty, xty1)
                y_sq_norm += y @ y

                del x, y, w, xtx1, xty1
                gc.collect()

        if scales is None:
            n_data = sum(
                [
                    len(dft_dict["energy"])
                    for dft_dict in self.multiple_dft_dicts.values()
                ]
            )
            variance = xe_sq_sum / n_data - np.square(xe_sum / n_data)
            self.scales = np.sqrt(variance)
        else:
            self.scales = scales

        xtx /= self.scales[:, np.newaxis]
        xtx /= self.scales[np.newaxis, :]
        xty /= self.scales

        self.reg_dict = dict()
        self.reg_dict["xtx"] = xtx
        self.reg_dict["xty"] = xty
        self.reg_dict["y_sq_norm"] = y_sq_norm
        self.reg_dict["total_n_data"] = total_n_data
        self.reg_dict["scales"] = self.scales

    def __find_min_energy(self):

        min_e = 1e10
        for _, dft_dict in self.multiple_dft_dicts.items():
            e_per_atom = dft_dict["energy"] / dft_dict["total_n_atoms"]
            min_e_trial = np.min(e_per_atom)
            if min_e_trial < min_e:
                min_e = min_e_trial
        return min_e

    def __sum_array(self, array1, array2):

        if array1 is None:
            return array2
        array1 += array2
        return array1

    def get_scales(self):
        return self.scales

    def get_updated_regression_dict(self):
        return self.reg_dict
