#!/usr/bin/env python
import gc

import numpy as np

from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.utils_sequential import get_batch_slice, slice_dft_dict
from pypolymlp.mlp_dev.core.utils_weights import apply_weight_percentage
from pypolymlp.mlp_dev.ensemble.mlpdev_dataxy_ensemble_base import (
    PolymlpDevDataXYEnsembleBase,
)


class PolymlpDevDataXYFeatureBagging(PolymlpDevDataXYEnsembleBase):

    def __init__(self, params: PolymlpDevData):

        super().__init__(params)

    def run(
        self,
        n_models=20,
        ratio_feature_samples=0.1,
        verbose=True,
        bootstrap_data=False,
    ):

        self.compute_features()
        self.apply_scales()
        self.apply_weights()

        print("Feature Bagging: Calculating X.T @ X")
        self.__sample(
            n_models=n_models,
            ratio_feature_samples=ratio_feature_samples,
            verbose=verbose,
        )
        if bootstrap_data:
            self.train_regression_dict_list = self.__compute_products_bootstrap(
                self.train_regression_dict,
            )
        else:
            self.train_regression_dict_list = self.__compute_products(
                self.train_regression_dict,
            )

        self.test_regression_dict_list = self.__compute_products(
            self.test_regression_dict
        )
        print("Finished")

        return self

    def compute_features(self):

        f_obj_train = self.features_class(self.params_dict, self.train_dict)
        f_obj_test = self.features_class(self.params_dict, self.test_dict)

        self.train_regression_dict = f_obj_train.regression_dict
        self.test_regression_dict = f_obj_test.regression_dict

        self.n_features = self.train_regression_dict["x"].shape[1]
        if self.is_hybrid:
            self.cumulative_n_features = f_obj_train.cumulative_n_features

        return self

    def __sample(self, n_models=20, ratio_feature_samples=0.1, verbose=True):

        self.n_models = n_models
        n_features_samples = round(self.n_features * ratio_feature_samples)
        self.random_indices = np.array(
            [
                np.random.choice(
                    range(self.n_features),
                    size=n_features_samples,
                    replace=False,
                )
                for _ in range(n_models)
            ]
        )

        if verbose:
            print("Size (X.T @ X):", n_features_samples, "** 2")
            print(
                "Estimated memory allocation (X.T @ X):",
                "{:.2f}".format(pow(n_features_samples, 2) * 8e-9 * n_models),
                "(GB)",
            )

        return self

    def __compute_products(self, reg_dict):

        print("Compute X.T @ X without bootstrapping dataset")
        x, y = reg_dict["x"], reg_dict["y"]
        reg_dict_array = []
        for i, r_indices in enumerate(self.random_indices):
            print("X.T @ X calculation: Model", i)
            x_samp = x[:, r_indices]
            y_samp = y
            reg_dict_add = {
                "xtx": x_samp.T @ x_samp,
                "xty": x_samp.T @ y_samp,
                "y_sq_norm": y_samp @ y_samp,
                "total_n_data": y_samp.shape[0],
                "scales": reg_dict["scales"][r_indices],
                "cumulative_n_features": self.cumulative_n_features,
            }
            reg_dict_array.append(reg_dict_add)

        return reg_dict_array

    def __compute_products_bootstrap(self, reg_dict):

        print("Compute X.T @ X with bootstrapping dataset")
        x, y = reg_dict["x"], reg_dict["y"]
        reg_dict_array = []
        for i, r_indices in enumerate(self.random_indices):
            print("X.T @ X calculation: Model", i)
            row_indices = np.random.choice(
                range(x.shape[0]),
                size=x.shape[0],
                replace=True,
            )
            x_samp = x[np.ix_(row_indices, r_indices)]
            y_samp = y[row_indices]
            reg_dict_add = {
                "xtx": x_samp.T @ x_samp,
                "xty": x_samp.T @ y_samp,
                "y_sq_norm": y_samp @ y_samp,
                "total_n_data": y_samp.shape[0],
                "scales": reg_dict["scales"][r_indices],
                "cumulative_n_features": self.cumulative_n_features,
            }
            reg_dict_array.append(reg_dict_add)

        return reg_dict_array


class PolymlpDevDataXYFeatureBaggingSequential(PolymlpDevDataXYEnsembleBase):

    def __init__(self, params: PolymlpDevData):

        super().__init__(params)

    def run(
        self,
        n_models=20,
        ratio_feature_samples=0.1,
        batch_size=64,
        verbose=True,
        element_swap=False,
    ):
        self.__sample(
            self.train_dict,
            n_models=n_models,
            ratio_feature_samples=ratio_feature_samples,
            verbose=verbose,
        )
        self.train_regression_dict_list = self.__compute_products(
            self.train_dict,
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap,
        )

        self.test_regression_dict_list = self.__compute_products(
            self.test_dict,
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap,
        )

        return self

    def __sample(
        self,
        dft_dicts,
        n_models=20,
        ratio_feature_samples=0.1,
        verbose=True,
    ):

        self.n_models = n_models
        self.n_features = self.__compute_n_features(dft_dicts)
        n_features_samples = round(self.n_features * ratio_feature_samples)
        self.random_indices = np.array(
            [
                np.random.choice(
                    range(self.n_features),
                    size=n_features_samples,
                    replace=False,
                )
                for _ in range(n_models)
            ]
        )

        if verbose:
            print("Size (X.T @ X):", n_features_samples, "** 2")
            print(
                "Estimated memory allocation (X.T @ X):",
                "{:.2f}".format(
                    n_features_samples * n_features_samples * 16e-9 * n_models
                ),
                "(GB)",
            )

        return self

    def __compute_n_features(self, dft_dicts):

        dft_dict = list(dft_dicts.values())[0]
        dft_dict_sliced = slice_dft_dict(dft_dict, 0, 2)
        dft_dict_tmp = {"tmp": dft_dict_sliced}
        features = self.features_class(
            self.params_dict,
            dft_dict_tmp,
            print_memory=False,
            element_swap=False,
        )
        if self.is_hybrid:
            self.cumulative_n_features = features.cumulative_n_features

        return features.x.shape[1]

    def __compute_products(
        self,
        dft_dicts,
        batch_size=64,
        verbose=True,
        element_swap=False,
    ):

        xtx = [None for _ in range(self.n_models)]
        xty = [None for _ in range(self.n_models)]
        y_sq_norm = [0.0 for _ in range(self.n_models)]
        xe_sum = [None for _ in range(self.n_models)]
        xe_sq_sum = [None for _ in range(self.n_models)]

        total_n_data = 0
        for set_id, dft_dict in dft_dicts.items():
            if verbose:
                print("----- Dataset:", set_id, "-----")

            n_str = len(dft_dict["structures"])
            begin_ids, end_ids = get_batch_slice(n_str, batch_size)
            for begin, end in zip(begin_ids, end_ids):
                if verbose:
                    print("Number of structures:", end - begin)

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

                n_data = x.shape[0]
                total_n_data += n_data
                for i, r_indices in enumerate(self.random_indices):
                    print("Batch x.T @ x calculations:", i)
                    x_samp = features.x[:, r_indices]
                    if self.scales_list is None:
                        xe = x_samp[:ne]
                        local1 = np.sum(xe, axis=0)
                        local2 = np.sum(np.square(xe), axis=0)
                        xe_sum[i] = self.__sum_array(xe_sum[i], local1)
                        xe_sq_sum[i] = self.__sum_array(xe_sq_sum[i], local2)

                    y = np.zeros(n_data)
                    w = np.ones(n_data)

                    x_samp, y, w = apply_weight_percentage(
                        x_samp,
                        y,
                        w,
                        dft_dict_sliced,
                        self.common_params_dict,
                        first_indices,
                        min_e=self.min_energy,
                    )
                    xtx[i] = self.__sum_array(xtx[i], x_samp.T @ x_samp)
                    xty[i] = self.__sum_array(xty[i], x_samp.T @ y)
                    y_sq_norm[i] += y @ y

                del x, y, w
                gc.collect()

        if self.scales_list is None:
            self.scales_list = []
            n_data = sum([len(d["energy"]) for d in dft_dicts.values()])
            for i in range(self.n_models):
                variance = xe_sq_sum[i] / n_data - np.square(xe_sum[i] / n_data)
                self.scales_list.append(np.sqrt(variance))

        reg_dict_array = []
        for i in range(self.n_models):
            scales = self.scales_list[i]
            xtx[i] /= scales[:, np.newaxis]
            xtx[i] /= scales[np.newaxis, :]
            xty[i] /= scales
            reg_dict = {
                "xtx": xtx[i],
                "xty": xty[i],
                "y_sq_norm": y_sq_norm[i],
                "total_n_data": total_n_data,
                "scales": scales,
                "cumulative_n_features": self.cumulative_n_features,
            }
            reg_dict_array.append(reg_dict)

        return reg_dict_array

    def __sum_array(self, array1, array2):

        if array1 is None:
            return array2
        array1 += array2
        return array1
