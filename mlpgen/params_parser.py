#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import mlpcpp
from polymlp_generator.common.input_parser import InputParser

class ParamsParser:

    def __init__(self, filename):

        self.parser = InputParser(filename)

        params = dict()
        params['n_type'] = self.parser.get_params('n_type', 
                                                  default=1, 
                                                  dtype=int)
        params['include_force'] = self.parser.get_params('include_force', 
                                                         default=True, 
                                                         dtype=bool)
        params['include_stress'] = self.parser.get_params('include_stress', 
                                                          default=True,  
                                                          dtype=bool)
        params['model'] = self.__get_potential_model_params(params['n_type'])


#        self.di.fmt_vaspruns = p.get_params('fmt_vaspruns', 
#                                            default='order', 
#                                            dtype=str)
#
        params['atomic_energy'] = self.__get_atomic_energy(params['n_type'])
        params['reg'] = self.__get_regression_params()
        params['dft'] = self.__get_vaspruns()

        self.params_dict = params

    def __get_potential_model_params(self, n_type):

        model = dict()
        model['cutoff'] = self.parser.get_params('cutoff',
                                                 default=6.0,
                                                 dtype=float)
        model['model_type'] = self.parser.get_params('model_type',
                                                     default=1,
                                                     dtype=int)
        model['max_p'] = self.parser.get_params('max_p',
                                                default=1,
                                                dtype=int)
        model['feature_type'] = self.parser.get_params('feature_type',
                                                       default='gtinv')

        params_gtinv = dict()
        if model['feature_type'] == 'gtinv':
            params_gtinv['order'] = self.parser.get_params('gtinv_order',
                                                           default=3,
                                                           dtype=int)
            size = params_gtinv['order'] - 1
            d_maxl = [2 for i in range(size)]
            params_gtinv['max_l'] = self.parser.get_params('gtinv_maxl',
                                                           size=size,
                                                           default=d_maxl,
                                                           dtype=int,
                                                           return_array=True)
            if len(params_gtinv['max_l']) < size:
                size_gap = size - len(params_gtinv['max_l'])
                for i in range(size_gap):
                    params_gtinv['max_l'].append(2)

            gtinv_sym = [False for i in range(size)]
            rgi = mlpcpp.Readgtinv(params_gtinv['order'],
                                   params_gtinv['max_l'],
                                   gtinv_sym,
                                   n_type)
            params_gtinv['lm_seq'] = rgi.get_lm_seq()
            params_gtinv['l_comb'] = rgi.get_l_comb()
            params_gtinv['lm_coeffs'] = rgi.get_lm_coeffs()
            model['max_l'] = max(params_gtinv['max_l'])
        else:
            params_gtinv['order'] = 0
            params_gtinv['max_l'] = []
            params_gtinv['lm_seq'] = []
            params_gtinv['l_comb'] = []
            params_gtinv['lm_coeffs'] = []
            model['max_l'] = 0
        model['gtinv'] = params_gtinv

        model['pair_type'] = 'gaussian'
        d_params1 = [1.0,1.0,1]
        params1 = self.parser.get_sequence('gaussian_params1', 
                                           default=d_params1)
        d_params2 = [0.0, model['cutoff']-1.0, 7]
        params2 = self.parser.get_sequence('gaussian_params2',
                                           default=d_params2)
        model['pair_params'] = list(itertools.product(params1, params2))
        model['pair_params'].append([0.0,0.0])

        return model

    def __get_atomic_energy(self, n_type):

        d_atom_e = [0.0 for i in range(n_type)]
        atom_e = self.parser.get_params('atomic_energy',
                                        size=n_type,
                                        default=d_atom_e,
                                        dtype=float,
                                        return_array=True)
        return atom_e

    def __get_regression_params(self):
        reg = dict()
        reg['method'] = 'ridge'
        d_alpha = [-3,1,5]
        reg['alpha'] = self.parser.get_sequence('reg_alpha_params', 
                                                default=d_alpha)
        return reg

    def get_params(self):
        return self.params_dict

    def __get_vaspruns(self):
        train = self.parser.get_params('train_data',default=None)
        test = self.parser.get_params('test_data',default=None)

        data = dict()
        data['train'] = glob.glob(train)
        data['test'] = glob.glob(test)
        return data
 

