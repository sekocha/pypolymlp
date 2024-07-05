#!/usr/bin/env python
import gc

import numpy as np

from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase
from pypolymlp.mlp_dev.core.utils_sequential import get_batch_slice, slice_dft_dict
from pypolymlp.mlp_dev.core.utils_weights import apply_weight_percentage


class PolymlpDevDataXY(PolymlpDevDataXYBase):

    def __init__(self, params: PolymlpDevData, verbose=True):
        """
        Keys in reg_dict
        ----------------
        - x
        - y
        - scales
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        """
        super().__init__(params, verbose=verbose)

    def run(self):

        self.compute_features()
        self.apply_scales()
        self.apply_weights()
        return self

    def compute_features(self):

        f_obj_train = self.features_class(
            self.params_dict,
            self.train_dict,
            print_memory=self.verbose,
        )
        f_obj_test = self.features_class(
            self.params_dict,
            self.test_dict,
            print_memory=self.verbose,
        )

        self.train_regression_dict = f_obj_train.regression_dict
        self.test_regression_dict = f_obj_test.regression_dict

        return self


class PolymlpDevDataXYSequential(PolymlpDevDataXYBase):

    def __init__(self, params: PolymlpDevData, verbose=True):
        """
        Keys in reg_dict
        ----------------
        - x.T @ X
        - x.T @ y
        - y_sq_norm,
        - scales
        - total_n_data,
        """
        super().__init__(params, verbose=verbose)

        if not self.is_multiple_datasets:
            raise ValueError(
                "Sequential version is available "
                "for PolymlpDevParams with multiple datasets."
            )

    def run(self, batch_size=64, verbose=True, element_swap=False):

        self.run_train(
            batch_size=batch_size, verbose=verbose, element_swap=element_swap
        )
        self.run_test(batch_size=batch_size, verbose=verbose, element_swap=element_swap)
        return self

    def run_train(self, batch_size=64, verbose=True, element_swap=False):
        self.train_regression_dict = self.compute_products(
            self.train_dict,
            scales=None,
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap,
        )
        return self

    def run_test(self, batch_size=64, verbose=True, element_swap=False):
        self.test_regression_dict = self.compute_products(
            self.test_dict,
            scales=self.__scales,
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap,
        )

    def clear_train(self):
        # del self.train_regression_dict
        self.train_regression_dict = None
        gc.collect()

    def clear_test(self):
        # del self.test_regression_dict
        self.test_regression_dict = None
        gc.collect()

    def compute_products(
        self,
        dft_dicts,
        scales=None,
        batch_size=64,
        verbose=True,
        element_swap=False,
    ):

        xtx, xty, y_sq_norm = None, None, 0.0
        xe_sum, xe_sq_sum = None, None
        total_n_data = 0
        for set_id, dft_dict in dft_dicts.items():
            if verbose:
                print("----- Dataset:", set_id, "-----")

            n_str = len(dft_dict["structures"])
            begin_ids, end_ids = get_batch_slice(n_str, batch_size)
            for begin, end in zip(begin_ids, end_ids):
                if verbose:
                    print("Structures:", end, "/", n_str)

                dft_dict_sliced = slice_dft_dict(dft_dict, begin, end)
                dft_dict_tmp = {"tmp": dft_dict_sliced}
                features = self.features_class(
                    self.params_dict,
                    dft_dict_tmp,
                    print_memory=verbose,
                    element_swap=element_swap,
                )
                x = features.x
                first_indices = features.first_indices[0]
                ne, nf, ns = features.regression_dict["n_data"]

                if verbose:
                    peak_mem = (
                        x.shape[0] * x.shape[1] + x.shape[1] * x.shape[1]
                    ) * 8e-9
                    print(
                        " Estimated peak memory allocation (X and X.T @ X):",
                        "{:.2f}".format(peak_mem),
                        "(GB)",
                    )

                if scales is None:
                    xe = x[:ne]
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
                    self.common_params_dict,
                    first_indices,
                    min_e=self.min_energy,
                )
                if verbose:
                    print("Compute X.T @ X")
                if n_features > 30000:
                    xtx = self.__sum_large_xtx(xtx, x)
                else:
                    xtx = self.__sum_array(xtx, x.T @ x)

                if verbose:
                    print("Compute X.T @ y")
                xty = self.__sum_array(xty, x.T @ y)
                y_sq_norm += y @ y

                del x, y, w
                gc.collect()

        if n_features > 30000:
            xtx = self.__large_transpose_xtx(xtx)

        if scales is None:
            n_data = sum([len(d["energy"]) for d in dft_dicts.values()])
            variance = xe_sq_sum / n_data - np.square(xe_sum / n_data)
            self.__scales = np.sqrt(variance)
        else:
            self.__scales = scales

        xtx /= self.__scales[:, np.newaxis]
        xtx /= self.__scales[np.newaxis, :]
        xty /= self.__scales

        reg_dict = {
            "xtx": xtx,
            "xty": xty,
            "y_sq_norm": y_sq_norm,
            "total_n_data": total_n_data,
            "scales": self.__scales,
        }
        if self.is_hybrid:
            reg_dict["cumulative_n_features"] = features.cumulative_n_features

        return reg_dict

    def __sum_array(self, array1, array2):

        if array1 is None:
            return array2
        array1 += array2
        return array1

    def __sum_large_xtx(self, xtx, x, n_batch=8):

        n_features = x.shape[1]
        if xtx is None:
            xtx = np.zeros((n_features, n_features))
            # xtx = x.T @ x
            # return xtx

        if n_features < n_batch:
            xtx += x.T @ x
        else:
            begin_ids, end_ids = get_batch_slice(n_features, n_features // n_batch)
            for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
                if self.verbose:
                    print("Batch:", end_row, "/", n_features)
                for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                    if i <= j:
                        xtx[begin_row:end_row, begin_col:end_col] += (
                            x[:, begin_row:end_row].T @ x[:, begin_col:end_col]
                        )
        return xtx

    def __large_transpose_xtx(self, xtx, n_batch=8):
        n_features = xtx.shape[0]
        begin_ids, end_ids = get_batch_slice(n_features, n_features // n_batch)
        for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
            for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                if i > j:
                    xtx[begin_row:end_row, begin_col:end_col] = xtx[
                        begin_col:end_col, begin_row:end_row
                    ].T
        return xtx
