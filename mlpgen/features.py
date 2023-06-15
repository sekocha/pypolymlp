#!/usr/bin/env python
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import mlpcpp

class Features:

    def __init__(self,
                 params_dict,
                 structures,
                 print_memory=True,
                 element_swap=False):  

        # for single dataset
        n_st_dataset = [len(structures)]
        force_dataset = [params_dict['include_force']]
        ##

        axis_array = [st['axis'] for st in structures]
        positions_c_array = [np.dot(st['axis'], st['positions']) 
                             for st in structures]
        types_array = [st['types'] for st in structures]
        n_atoms_sum_array = [sum(st['n_atoms']) for st in structures]

        obj = mlpcpp.PotentialModel(axis_array, 
                                    positions_c_array, 
                                    types_array, 
                                    params_dict['n_type'], 
                                    False, 
                                    params_dict['model']['pair_params'], 
                                    params_dict['model']['cutoff'], 
                                    params_dict['model']['pair_type'], 
                                    params_dict['model']['feature_type'], 
                                    params_dict['model']['model_type'], 
                                    params_dict['model']['max_p'], 
                                    params_dict['model']['max_l'], 
                                    params_dict['model']['gtinv']['lm_seq'], 
                                    params_dict['model']['gtinv']['l_comb'], 
                                    params_dict['model']['gtinv']['lm_coeffs'], 
                                    n_st_dataset, 
                                    force_dataset, 
                                    n_atoms_sum_array,
                                    print_memory,
                                    element_swap)

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


