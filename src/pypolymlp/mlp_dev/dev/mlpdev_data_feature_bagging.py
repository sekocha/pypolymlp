#!/usr/bin/env python 
import numpy as np
import gc

from pypolymlp.mlp_dev.mlpdev_core import PolymlpDevParams
from pypolymlp.mlp_dev.mlpdev_data_base import PolymlpDevBase

from pypolymlp.mlp_dev.sequential import (
    get_batch_slice, slice_dft_dict,
)
from pypolymlp.mlp_dev.weights import apply_weight_percentage


class PolymlpDevFeatureBagging(PolymlpDevBase):

    def __init__(self, params: PolymlpDevParams):
        """
        Keys in reg_dict
        ----------------
        - x.T @ X
        - x.T @ y
        - y_sq_norm,
        - scales
        - total_n_data,
        """
        super().__init__(params)

        if not self.is_multiple_datasets:
            raise ValueError('Ensemble version is available '
                             'for PolymlpDevParams with multiple datasets.')

        self.__random_indices = None
        self.__n_models = None
        self.__n_features = None
        self.__train_regression_dict_list = None
        self.__test_regression_dict_list = None

    def run(self, n_models=20, ratio_feature_samples=0.1):

        self.compute_features()
        self.apply_scales()
        self.apply_weights()

        print('Feature Bagging: Calculating X.T @ X')
        self.__train_regression_dict_list = self.compute_products(
            self.train_regression_dict,
            n_models=n_models,
            ratio_feature_samples=ratio_feature_samples,
        )
        self.__test_regression_dict_list = self.compute_products(
            self.test_regression_dict
        )
        print('Finished')

        return self

    def compute_features(self):

        f_obj_train = self.features_class(self.params_dict, self.train_dict)
        f_obj_test = self.features_class(self.params_dict, self.test_dict)

        self.train_regression_dict = f_obj_train.regression_dict
        self.test_regression_dict = f_obj_test.regression_dict

        if self.is_hybrid:
            self.__cumulative_n_features = features.cumulative_n_features

        self.__n_features = self.train_regression_dict['x'].shape[1]
        return self

    def compute_products(
            self, reg_dict, n_models=20, ratio_feature_samples=0.1,
        ):

        if self.__random_indices is None:
            n_features_samples = round(
                self.__n_features * ratio_feature_samples
            )
            self.__random_indices = np.array(
                [
                    np.random.choice(
                        range(self.__n_features), 
                        size=n_features_samples, 
                        replace=False,
                    ) for _ in range(n_models)
                ]
            )
        self.__n_models = self.__random_indices.shape[0]

        x, y = reg_dict['x'], reg_dict['y']
        reg_dict_array = []
        for i, r_indices in enumerate(self.__random_indices):
            x_samp = x[:,r_indices]
            reg_dict_add = {
                'xtx': x_samp.T @ x_samp,
                'xty': x_samp.T @ y,
                'y_sq_norm': y @ y,
                'total_n_data': y.shape[0],
                'scales': reg_dict['scales'][r_indices],
            }
            if self.is_hybrid:
                reg_dict_add['cumulative_n_features'] \
                    = self.__cumulative_n_features

            reg_dict_array.append(reg_dict_add)

        return reg_dict_array

    @property
    def random_indices(self):
        return self.__random_indices

    @property
    def n_models(self):
        return self.__n_models

    @property
    def n_features(self):
        return self.__n_features

    @property
    def train_regression_dict_list(self):
        return self.__train_regression_dict_list

    @property
    def test_regression_dict_list(self):
        return self.__test_regression_dict_list


class PolymlpDevFeatureBaggingSequential(PolymlpDevBase):

    def __init__(self, params: PolymlpDevParams):
        """
        Keys in reg_dict
        ----------------
        - x.T @ X
        - x.T @ y
        - y_sq_norm,
        - scales
        - total_n_data,
        """
        super().__init__(params)

        if not self.is_multiple_datasets:
            raise ValueError('Ensemble version is available '
                             'for PolymlpDevParams with multiple datasets.')

        self.__random_indices = None
        self.__n_models = None
        self.__n_features = None
        self.__train_regression_dict_list = None
        self.__test_regression_dict_list = None
        self.__scales_array = None

    def run(
        self, n_models=20, ratio_feature_samples=0.1,
        batch_size=64, verbose=True, element_swap=False
    ):

        self.__train_regression_dict_list = self.compute_products(
            self.train_dict, 
            n_models=n_models,
            ratio_feature_samples=ratio_feature_samples,
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap
        )

        self.__test_regression_dict_list = self.compute_products(
            self.test_dict, 
            batch_size=batch_size,
            verbose=verbose,
            element_swap=element_swap
        )

        return self

    def compute_n_features(self, dft_dicts):

        dft_dict = list(dft_dicts.values())[0]
        dft_dict_sliced = slice_dft_dict(dft_dict, 0, 2)
        dft_dict_tmp = {'tmp': dft_dict_sliced}
        features = self.features_class(
            self.params_dict, 
            dft_dict_tmp,
            print_memory=False,
            element_swap=False,
        )
        return features.x.shape[1]

    def compute_products(
        self, dft_dicts, 
        n_models=20, ratio_feature_samples=0.1,
        batch_size=64, verbose=True, element_swap=False,
    ):

        if self.__random_indices is None:
            self.__n_features = self.compute_n_features(dft_dicts)
            n_features_samples = round(
                self.__n_features * ratio_feature_samples
            )
            self.__random_indices = np.array(
                [
                    np.random.choice(
                        range(self.__n_features), 
                        size=n_features_samples, 
                        replace=False,
                    ) for _ in range(n_models)
                ]
            )

            if verbose:
                print(' Estimated memory allocation (X.T @ X):', 
                    '{:.2f}'.format(
                        n_features_samples * n_features_samples 
                        * 16e-9 * n_models
                    ), '(GB)')

        self.__n_models = n_models = self.__random_indices.shape[0]

        xtx = [None for _ in range(n_models)]
        xty = [None for _ in range(n_models)]
        y_sq_norm = [0.0 for _ in range(n_models)]
        xe_sum = [None for _ in range(n_models)]
        xe_sq_sum = [None for _ in range(n_models)]

        total_n_data = 0
        for set_id, dft_dict in dft_dicts.items():
            if verbose:
                print('----- Dataset:', set_id, '-----')

            n_str = len(dft_dict['structures'])
            begin_ids, end_ids = get_batch_slice(n_str, batch_size)
            for begin, end in zip(begin_ids, end_ids):
                if verbose:
                    print('Number of structures:', end - begin)

                dft_dict_sliced = slice_dft_dict(dft_dict, begin, end)
                dft_dict_tmp = {'tmp': dft_dict_sliced}
                features = self.features_class(
                    self.params_dict, 
                    dft_dict_tmp,
                    print_memory=verbose,
                    element_swap=element_swap
                )
                x = features.x
                first_indices = features.first_indices[0]
                ne, nf, ns = features.regression_dict['n_data']

                n_data = x.shape[0]
                total_n_data += n_data
                for i, r_indices in enumerate(self.__random_indices):
                    print('small x.T @ x calculations:')
                    print('size (x.T @ x):', 
                          r_indices.shape[0], r_indices.shape[0])
                    x_samp = features.x[:,r_indices]
                    if self.__scales_array is None:
                        xe = x_samp[:ne]
                        local1 = np.sum(xe, axis=0)
                        local2 = np.sum(np.square(xe), axis=0)
                        xe_sum[i] = self.__sum_array(xe_sum[i], local1)
                        xe_sq_sum[i] = self.__sum_array(xe_sq_sum[i], local2)

                    y = np.zeros(n_data)
                    w = np.ones(n_data)

                    x_samp, y, w = apply_weight_percentage(
                                        x_samp, y, w,
                                        dft_dict_sliced,
                                        self.common_params_dict,
                                        first_indices,
                                        min_e=self.min_energy
                                    )
                    xtx[i] = self.__sum_array(xtx[i], x_samp.T @ x_samp)
                    xty[i] = self.__sum_array(xty[i], x_samp.T @ y)
                    y_sq_norm[i] += y @ y

                del x, y, w
                gc.collect()

        if self.__scales_array is None:
            self.__scales_array = []
            n_data = sum(
                [len(d['energy']) for d in dft_dicts.values()]
            )

            for i in range(n_models):
                variance = xe_sq_sum[i] / n_data - np.square(xe_sum[i] / n_data)
                self.__scales_array.append(np.sqrt(variance))

        reg_dict_array = []
        for i in range(n_models):
            xtx[i] /= self.__scales_array[i][:, np.newaxis]
            xtx[i] /= self.__scales_array[i][np.newaxis, :]
            xty[i] /= self.__scales_array[i]

            reg_dict = {
                'xtx': xtx[i],
                'xty': xty[i],
                'y_sq_norm': y_sq_norm[i],
                'total_n_data': total_n_data,
                'scales': self.__scales_array[i],
            }
            if self.is_hybrid:
                reg_dict['cumulative_n_features'] \
                    = features.cumulative_n_features

            reg_dict_array.append(reg_dict)

        return reg_dict_array

    def __sum_array(self, array1, array2):

        if array1 is None:
            return array2
        array1 += array2
        return array1

    @property
    def random_indices(self):
        return self.__random_indices

    @property
    def n_models(self):
        return self.__n_models

    @property
    def n_features(self):
        return self.__n_features

    @property
    def train_regression_dict_list(self):
        return self.__train_regression_dict_list

    @property
    def test_regression_dict_list(self):
        return self.__test_regression_dict_list

