#!/usr/bin/env python
import numpy as np
import gc

from pypolymlp.mlp_gen.multi_datasets.features import Features
from pypolymlp.mlp_gen.precondition import apply_atomic_energy
from pypolymlp.mlp_gen.precondition import apply_weight_percentage

from pypolymlp.mlp_gen.multi_datasets.sequential import (
    get_batch_slice, slice_dft_dict,
)

class PreconditionMultiDatasets:
    
    def __init__(self, params_dict, multiple_dft_dicts):

        self.__params_dict = params_dict
        self.__multiple_dft_dicts = multiple_dft_dicts

    def find_min_energy(self):

        min_e = 1e10
        for _, dft_dict in self.__multiple_dft_dicts.items():
            e_per_atom = dft_dict['energy'] / dft_dict['total_n_atoms']
            min_e_trial = np.min(e_per_atom)
            if min_e_trial < min_e:
                min_e = min_e_trial
        return min_e

    def apply_atomic_energy(self):

        for _, dft_dict in self.__multiple_dft_dicts.items():
            dft_dict = apply_atomic_energy(dft_dict, self.__params_dict)
        return self.__multiple_dft_dicts


class Ensemble:

    def __init__(self, params_dict, multiple_dft_dicts):  

        self.__params_dict = params_dict
        precond = PreconditionMultiDatasets(params_dict, multiple_dft_dicts)
        self.__multiple_dft_dicts = precond.apply_atomic_energy()
        self.__min_e_per_atom = precond.find_min_energy()

        self.__scales = None
        self.__reg_dict = dict()

    def run(self, scales=None, verbose=True, element_swap=False, batch_size=64):

        xtx, xty, y_sq_norm = None, None, 0.0
        xe_sum, xe_sq_sum = None, None
        total_n_data = 0
        for set_id, dft_dict in self.__multiple_dft_dicts.items():
            if verbose:
                print('----- Dataset:', set_id, '-----')

            n_str = len(dft_dict['structures'])
            begin_ids, end_ids = get_batch_slice(n_str, batch_size)
            for begin, end in zip(begin_ids, end_ids):
                if verbose:
                    print('Number of structures:', end - begin)

                dft_dict_sliced = slice_dft_dict(dft_dict, begin, end)
                dft_dict_tmp = {'tmp': dft_dict_sliced}
                features = Features(
                    self.__params_dict, 
                    dft_dict_tmp, 
                    print_memory=verbose, 
                    element_swap=element_swap
                )
                x = features.get_x()
                first_indices = features.get_first_indices()[0]

                if scales is None:
                    xe = x[:features.ne]
                    local1 = np.sum(xe, axis=0)
                    local2 = np.sum(np.square(xe), axis=0)
                    xe_sum = self.__sum_array(xe_sum, local1)
                    xe_sq_sum = self.__sum_array(xe_sq_sum, local2)

                n_data, n_features = x.shape
                y = np.zeros(n_data)
                w = np.ones(n_data)
                total_n_data += n_data

                x, y, w = apply_weight_percentage(
                                x, y, w, 
                                dft_dict_sliced, 
                                self.__params_dict, 
                                first_indices,
                                min_e=self.__min_e_per_atom
                          )
                xtx = self.__sum_array(xtx, x.T @ x)
                xty = self.__sum_array(xty, x.T @ y)
                y_sq_norm += y @ y

                del x, y, w
                gc.collect()

        if scales is None:
            n_data = sum(
                [len(d['energy']) for d in self.__multiple_dft_dicts.values()]
            )
            variance = xe_sq_sum / n_data - np.square(xe_sum / n_data)
            self.__scales = np.sqrt(variance)
        else:
            self.__scales = scales

        xtx /= self.__scales[:, np.newaxis]
        xtx /= self.__scales[np.newaxis, :]
        xty /= self.__scales

        self.__reg_dict['xtx'] = xtx
        self.__reg_dict['xty'] = xty
        self.__reg_dict['y_sq_norm'] = y_sq_norm
        self.__reg_dict['total_n_data'] = total_n_data
        self.__reg_dict['scales'] = self.__scales

        return self
            
    def __sum_array(self, array1, array2):

        if array1 is None:
            return array2
        array1 += array2
        return array1

    @property
    def scales(self):
        return self.__scales

    @property
    def regression_dict(self):
        return self.__reg_dict

