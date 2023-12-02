#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import itertools
from collections import defaultdict
from distutils.util import strtobool

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../c++/lib')
import mlpcpp
from pypolymlp.common.input_parser import InputParser
from pypolymlp.common.vasp import Vasprun

class ParamsParser:

    def __init__(self, filename, 
                 multiple_datasets=False, 
                 parse_vasprun_locations=True):

        self.parser = InputParser(filename)

        params = dict()
        params['n_type'] = self.parser.get_params('n_type', 
                                                  default=1, 
                                                  dtype=int)
        params['include_force'] = self.parser.get_params('include_force', 
                                                         default=True, 
                                                         dtype=bool)
        if params['include_force']:
            params['include_stress'] = self.parser.get_params('include_stress', 
                                                              default=True,  
                                                              dtype=bool)
        else:
            params['include_stress'] = False

        self.n_type = params['n_type']
        self.include_force = params['include_force']

        params['model'] = self.__get_potential_model_params(params['n_type'])
        params['atomic_energy'] = self.__get_atomic_energy(params['n_type'])
        params['reg'] = self.__get_regression_params()

        if parse_vasprun_locations:
            if multiple_datasets:
                params['dft'] = self.__get_multiple_vasprun_sets()
            else:
                params['dft'] = self.__get_single_vasprun_set()

        params['elements'] = self.parser.get_params('elements',
                                                     size=params['n_type'],
                                                     default=None,
                                                     required=True,
                                                     dtype=str,
                                                     return_array=True)
        rearrange = self.parser.get_params('rearrange_by_elements',
                                            default=True,
                                            dtype=bool)
        params['element_order'] = params['elements'] if rearrange else None

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

        gtinv_dict = dict()
        if model['feature_type'] == 'gtinv':
            gtinv_dict['order'] = self.parser.get_params('gtinv_order',
                                                         default=3,
                                                         dtype=int)
            size = gtinv_dict['order'] - 1
            d_maxl = [2 for i in range(size)]
            gtinv_dict['max_l'] = self.parser.get_params('gtinv_maxl',
                                                         size=size,
                                                         default=d_maxl,
                                                         dtype=int,
                                                         return_array=True)
            if len(gtinv_dict['max_l']) < size:
                size_gap = size - len(gtinv_dict['max_l'])
                for i in range(size_gap):
                    gtinv_dict['max_l'].append(2)

            gtinv_sym = [False for i in range(size)]
            rgi = mlpcpp.Readgtinv(gtinv_dict['order'],
                                   gtinv_dict['max_l'],
                                   gtinv_sym,
                                   n_type)
            gtinv_dict['lm_seq'] = rgi.get_lm_seq()
            gtinv_dict['l_comb'] = rgi.get_l_comb()
            gtinv_dict['lm_coeffs'] = rgi.get_lm_coeffs()
            model['max_l'] = max(gtinv_dict['max_l'])
        else:
            gtinv_dict['order'] = 0
            gtinv_dict['max_l'] = []
            gtinv_dict['lm_seq'] = []
            gtinv_dict['l_comb'] = []
            gtinv_dict['lm_coeffs'] = []
            model['max_l'] = 0
        model['gtinv'] = gtinv_dict

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

    def __get_single_vasprun_set(self):

        train = self.parser.get_params('train_data',default=None)
        test = self.parser.get_params('test_data',default=None)

        data = dict()
        data['train'] = sorted(glob.glob(train))
        data['test'] = sorted(glob.glob(test))
        return data

    def __get_multiple_vasprun_sets(self):

        train = self.parser.get_train()
        test = self.parser.get_test()

        for params in train:
            shortage = []
            if len(params) < 2:
                shortage.append('True')
            if len(params) < 3:
                shortage.append(1.0)
            params.extend(shortage)

        for params in test:
            shortage = []
            if len(params) < 2:
                shortage.append('True')
            if len(params) < 3:
                shortage.append(1.0)
            params.extend(shortage)

        if self.include_force == False:
            for params in train:
                params[1] = 'False'
            for params in test:
                params[1] = 'False'

        data = dict()
        data['train'], data['test'] = dict(), dict()
        for params in train:
            set_id = params[0]
            data['train'][set_id] = dict()
            data['train'][set_id]['vaspruns'] = sorted(glob.glob(set_id))
            data['train'][set_id]['include_force'] = strtobool(params[1])
            data['train'][set_id]['weight'] = float(params[2])
        for params in test:
            set_id = params[0]
            data['test'][set_id] = dict()
            data['test'][set_id]['vaspruns'] = sorted(glob.glob(set_id))
            data['test'][set_id]['include_force'] = strtobool(params[1])
            data['test'][set_id]['weight'] = float(params[2])
        return data

    def get_params(self):
        return self.params_dict

def permute_atoms(st, force, element_order):

    positions, n_atoms, elements, types = [], [], [], []
    force_permute = []
    for atomtype, ele in enumerate(element_order):
        ids = np.where(np.array(st['elements']) == ele)[0]
        n_match = len(ids)
        positions.extend(st['positions'][:,ids].T)
        n_atoms.append(n_match)
        elements.extend([ele for _ in range(n_match)])
        types.extend([atomtype for _ in range(n_match)])
        force_permute.extend(force[:,ids].T)
    positions = np.array(positions).T
    force_permute = np.array(force_permute).T

    st['positions'] = positions
    st['n_atoms'] = n_atoms
    st['elements'] = elements
    st['types'] = types
    return st, force_permute

def parse_vaspruns(vaspruns, element_order=None):

    kbar_to_eV = 1 / 1602.1766208
    dft_dict = defaultdict(list)
    for vasp in vaspruns:
        v = Vasprun(vasp)
        property_dict = v.get_properties()
        structure_dict = v.get_structure()

        if element_order is not None:
            structure_dict, property_dict['force'] \
                    = permute_atoms(structure_dict, 
                                    property_dict['force'], 
                                    element_order)

        dft_dict['energy'].append(property_dict['energy'])
        force_ravel = np.ravel(property_dict['force'], order='F')
        dft_dict['force'].extend(force_ravel)

        sigma = property_dict['stress'] * structure_dict['volume'] * kbar_to_eV
        s = [sigma[0][0], sigma[1][1], sigma[2][2],
             sigma[0][1], sigma[1][2], sigma[2][0]]
        dft_dict['stress'].extend(s)
        dft_dict['structures'].append(structure_dict)

    dft_dict['energy'] = np.array(dft_dict['energy'])
    dft_dict['force'] = np.array(dft_dict['force'])
    dft_dict['stress'] = np.array(dft_dict['stress'])

    elements_size = [len(st['elements']) for st in dft_dict['structures']]
    elements = dft_dict['structures'][np.argmax(elements_size)]['elements']
    dft_dict['elements'] = sorted(set(elements), key=elements.index)

    dft_dict['total_n_atoms'] = np.array([sum(st['n_atoms'])
                                         for st in dft_dict['structures']])
    dft_dict['filenames'] = vaspruns

    return dft_dict


