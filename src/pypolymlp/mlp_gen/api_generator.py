#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import glob
import itertools

from pypolymlp.mlp_gen.generator import (
    run_generator_single_dataset,
    run_generator_single_dataset_from_params
)
from pypolymlp.cxx.lib import libmlpcpp

"""
    Variables in params_dict:
      - n_type
      - include_force
      - include_stress
      - model
        - cutoff
        - model_type
        - max_p
        - max_l
        - feature_type
        - pair_type
        - pair_params
        - gtinv
          - order
          - max_l
          - lm_seq
          - l_comb
          - lm_coeffs
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train (dataset locations)
        - test (dataset locations)

"""

class Pypolymlp:
    
    def __init__(self):

        self._params_dict = dict()

        self._params_dict['model'] = dict()
        self._params_dict['model']['gtinv'] = dict()

        self._params_dict['reg'] = dict()

        self._params_dict['dft'] = dict()
        self._params_dict['dft']['train'] = dict()
        self._params_dict['dft']['test'] = dict()

        self.train_dft_dict = None
        self.test_dft_dict = None

        self._mlp_dict = None

    def set_params(
            self, 
            elements,
            include_force=True,
            include_stress=False,
            cutoff=6.0,
            model_type=4,
            max_p=2,
            feature_type='gtinv',
            gaussian_params1=[1.0,1.0,1],
            gaussian_params2=[0.0,5.0,7],
            reg_alpha_params=[-3.0,1.0,5],
            gtinv_order=3,
            gtinv_maxl=[4,4,2,1,1],
            atomic_energy=None,
            rearrange_by_elements=True,
        ):

        self._params_dict['elements'] = elements
        n_type = len(elements)
        self._params_dict['n_type'] = n_type
        self._params_dict['include_force'] = include_force
        self._params_dict['include_stress'] = include_stress

        model = self._params_dict['model']
        model['cutoff'] = cutoff

        if model_type > 4:
            raise ValueError('model_type != 1, 2, 3, or 4')
        model['model_type'] = model_type
            
        if max_p > 3:
            raise ValueError('model_type != 1, 2, or 3')
        model['max_p'] = max_p

        if feature_type != 'gtinv' and feature_type != 'pair':
            raise ValueError('feature_type != gtinv or pair')
        model['feature_type'] = feature_type

        model['pair_type'] = 'gaussian'
        if len(gaussian_params1) != 3:
            raise ValueError('len(gaussian_params1) != 3')
        if len(gaussian_params2) != 3:
            raise ValueError('len(gaussian_params2) != 3')
        params1 = self.__sequence(gaussian_params1)
        params2 = self.__sequence(gaussian_params2)
        model['pair_params'] = list(itertools.product(params1, params2))
        model['pair_params'].append([0.0,0.0])

        gtinv_dict = self._params_dict['model']['gtinv']
        if model['feature_type'] == 'gtinv':
            gtinv_dict['order'] = gtinv_order
            size = gtinv_dict['order'] - 1
            if len(gtinv_maxl) < size:
                raise ValueError('size (gtinv_maxl) !=', size)
            gtinv_dict['max_l'] = gtinv_maxl
            gtinv_sym = [False for i in range(size)]
            rgi = libmlpcpp.Readgtinv(gtinv_dict['order'],
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

        if len(reg_alpha_params) != 3:
            raise ValueError('len(reg_alpha_params) != 3')
        self._params_dict['reg']['method'] = 'ridge'
        self._params_dict['reg']['alpha'] = self.__sequence(reg_alpha_params)

        if atomic_energy is None:
            self._params_dict['atomic_energy'] = [0.0 for i in range(n_type)]
        else:
            if len(atomic_energy) != n_type:
                raise ValueError('len(atomic_energy) != n_type')
            self._params_dict['atomic_energy'] = atomic_energy

        if rearrange_by_elements:
            self._params_dict['element_order'] = elements 
        else:
            self._params_dict['element_order'] = None

    def set_datasets_vasp(self, train_vaspruns, test_vaspruns):

        self._params_dict['dataset_type'] = 'vasp'
        self._params_dict['dft']['train'] = sorted(train_vaspruns)
        self._params_dict['dft']['test'] = sorted(test_vaspruns)

    def set_datasets_phono3py(
            self, 
            train_yaml, 
            test_yaml, 
            train_energy_dat, 
            test_energy_dat
    ):

        self._params_dict['dataset_type'] = 'phono3py'
        data = self._params_dict['dft']
        data['train'], data['test'] = dict(), dict()
        data['train']['phono3py_yaml'] = train_yaml
        data['train']['energy'] = train_energy_dat
        data['test']['phono3py_yaml'] = test_yaml
        data['test']['energy'] = test_energy_dat


    def set_datasets_displacements(disps, forces, energies):
        # train = self.train_dft_dict
        # test = self.test_dft_dict
        pass

    def __sequence(self, params):
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    def run(self, file_params=None, log=True):

        if file_params is None:
            self._mlp_dict = run_generator_single_dataset_from_params(
                self._params_dict,
                log=log,
            )
        else:
            self._mlp_dict = run_generator_single_dataset(file_params, log=log)

    @ property
    def parameters(self):
        return self._params_dict

    @ property
    def coeffs(self):
        return self._mlp_dict['coeffs'] / self._mlp_dict['scales']

    @ property
    def summary(self):
        return self._mlp_dict

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    polymlp = Pypolymlp()

    ''' from parameters'''
    polymlp.set_params(
        ['Mg','O'],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4,4],
        gaussian_params2=[0.0,7.0,8],
        atomic_energy=[-0.00040000,-1.85321219],
    )
    #print(polymlp.parameters)

    train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
    test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
    polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
    polymlp.run(log=True)

    ''' from polymlp.in'''
    #polymlp.run(file_params='polymlp.in', log=True)
