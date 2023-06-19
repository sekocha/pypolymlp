#!/usr/bin/env python
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import mlpcpp

class Features:

    def __init__(self,
                 params_dict,
                 multiple_dft_dicts, 
                 print_memory=True,
                 element_swap=False):  

        n_st_dataset, force_dataset = [], []
        axis_array, positions_c_array = [], []
        types_array, n_atoms_sum_array = [], []

        for _, dft_dict in multiple_dft_dicts.items():
            structures = dft_dict['structures']
            n_st_dataset.append(len(structures))
            force_dataset.append(dft_dict['include_force'])
            axis_array.extend([st['axis'] for st in structures])
            positions_c_array.extend([np.dot(st['axis'], st['positions']) 
                                      for st in structures])
            types_array.extend([st['types'] for st in structures])
            n_atoms_sum_array.extend([sum(st['n_atoms']) for st in structures])

        params_dict['element_swap'] = element_swap
        params_dict['print_memory'] = print_memory
        obj = mlpcpp.PotentialModel(params_dict,
                                    axis_array, 
                                    positions_c_array, 
                                    types_array, 
                                    n_st_dataset, 
                                    force_dataset, 
                                    n_atoms_sum_array)
 
        self.x = obj.get_x()
        self.fbegin, self.sbegin = obj.get_fbegin(), obj.get_sbegin()

        self.ebegin, ei = [], 0
        for n in n_st_dataset:
            self.ebegin.append(ei)
            ei += n
        self.ebegin = np.array(self.ebegin)

    def get_x(self):
        return self.x
    def get_ebegin(self):
        return self.ebegin
    def get_fbegin(self):
        return self.fbegin
    def get_sbegin(self):
        return self.sbegin
    def get_first_indices(self):
        return list(zip(self.ebegin, self.fbegin, self.sbegin))


